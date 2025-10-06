using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.M5
{
    public class M5VolumeFeatures : BaseFeatureCalculator
    {
        public override string Name => "M5_Volume";
        public override string Category => "Volume";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(5);
        public override int Priority => 25;

        private readonly RollingWindow<double> _volumeValues = new(50);
        private readonly RollingWindow<double> _obv = new(50);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 1) return;

            var bar = bars[currentIndex];
            var volume = (double)bar.TickVolume;

            // ===== BASIC VOLUME =====
            output.AddFeature("m5_volume", volume);
            _volumeValues.Add(volume);

            if (currentIndex >= 20)
            {
                var volumeMA = 0.0;
                for (int i = currentIndex - 19; i <= currentIndex; i++)
                {
                    volumeMA += bars[i].TickVolume;
                }
                volumeMA /= 20;

                output.AddFeature("m5_volume_ma", volumeMA);
                output.AddFeature("m5_volume_ratio", SafeDiv(volume, volumeMA));
            }

            // Volume Trend
            if (_volumeValues.Count >= 10)
            {
                var recent = _volumeValues.GetValues().Take(5).Average();
                var older = _volumeValues.GetValues().Skip(5).Take(5).Average();
                var trend = SafeDiv(recent - older, older);
                output.AddFeature("m5_volume_trend", trend);
            }
            else
            {
                output.AddFeature("m5_volume_trend", 0.0);
            }

            // ===== ON-BALANCE VOLUME (OBV) =====
            if (currentIndex >= 1)
            {
                var prevOBV = _obv.Count > 0 ? _obv[0] : 0;
                var currentOBV = prevOBV;

                if (bar.Close > bars[currentIndex - 1].Close)
                    currentOBV += volume;
                else if (bar.Close < bars[currentIndex - 1].Close)
                    currentOBV -= volume;

                _obv.Add(currentOBV);
                output.AddFeature("m5_obv", currentOBV / 1000);

                // OBV MA
                if (_obv.Count >= 10)
                {
                    var obvMA = _obv.GetValues().Take(10).Average();
                    output.AddFeature("m5_obv_ma", obvMA / 1000);

                    // OBV Divergence
                    if (currentIndex >= 10)
                    {
                        var priceTrend = (double)(bar.Close - bars[currentIndex - 10].Close);
                        var obvTrend = currentOBV - _obv[9];
                        var divergence = (priceTrend > 0 && obvTrend < 0) ? -1.0 :
                                       (priceTrend < 0 && obvTrend > 0) ? 1.0 : 0.0;
                        output.AddFeature("m5_obv_divergence", divergence);
                    }
                    else
                    {
                        output.AddFeature("m5_obv_divergence", 0.0);
                    }
                }
                else
                {
                    output.AddFeature("m5_obv_ma", 0.0);
                    output.AddFeature("m5_obv_divergence", 0.0);
                }
            }
            else
            {
                output.AddFeature("m5_obv", 0.0);
                output.AddFeature("m5_obv_ma", 0.0);
                output.AddFeature("m5_obv_divergence", 0.0);
            }

            // ===== VOLUME PRICE TREND (VPT) =====
            if (currentIndex >= 1)
            {
                var priceChange = SafeDiv(
                    (double)(bar.Close - bars[currentIndex - 1].Close),
                    (double)bars[currentIndex - 1].Close);
                var vpt = volume * priceChange;
                output.AddFeature("m5_vpt", vpt);
            }

            // ===== POSITIVE/NEGATIVE VOLUME INDEX =====
            if (currentIndex >= 1)
            {
                var prevVolume = (double)bars[currentIndex - 1].TickVolume;

                // PVI - changes when volume increases
                if (volume > prevVolume)
                {
                    var priceChange = SafeDiv(
                        (double)(bar.Close - bars[currentIndex - 1].Close),
                        (double)bars[currentIndex - 1].Close);
                    output.AddFeature("m5_pvi", priceChange * 100);
                }
                else
                {
                    output.AddFeature("m5_pvi", 0);
                }

                // NVI - changes when volume decreases
                if (volume < prevVolume)
                {
                    var priceChange = SafeDiv(
                        (double)(bar.Close - bars[currentIndex - 1].Close),
                        (double)bars[currentIndex - 1].Close);
                    output.AddFeature("m5_nvi", priceChange * 100);
                }
                else
                {
                    output.AddFeature("m5_nvi", 0);
                }
            }

            // ===== MONEY FLOW INDEX (MFI) =====
            if (currentIndex >= 14)
            {
                var mfi = CalculateMFI(bars, 14, currentIndex);
                output.AddFeature("m5_mfi", mfi);
            }

            // ===== CHAIKIN MONEY FLOW (CMF) =====
            if (currentIndex >= 20)
            {
                var cmf = CalculateCMF(bars, 20, currentIndex);
                output.AddFeature("m5_cmf", cmf);
            }

            // ===== EASE OF MOVEMENT (EMV) =====
            if (currentIndex >= 14)
            {
                var emv = CalculateEMV(bars, 14, currentIndex);
                output.AddFeature("m5_emv", emv);
            }

            // ===== VOLUME FORCE =====
            if (currentIndex >= 1)
            {
                var forceIndex = volume * (double)(bar.Close - bars[currentIndex - 1].Close) * 10000;
                output.AddFeature("m5_volume_force", forceIndex);
            }

            // ===== ACCUMULATION/DISTRIBUTION =====
            if (currentIndex >= 1)
            {
                var moneyFlowMultiplier = SafeDiv(
                    (double)((bar.Close - bar.Low) - (bar.High - bar.Close)),
                    (double)(bar.High - bar.Low));
                var moneyFlowVolume = moneyFlowMultiplier * volume;
                output.AddFeature("m5_accumulation_distribution", moneyFlowVolume / 1000);
            }

            // ===== VOLUME OSCILLATOR =====
            if (currentIndex >= 20)
            {
                var shortMA = 0.0;
                var longMA = 0.0;

                for (int i = currentIndex - 4; i <= currentIndex; i++)
                    shortMA += bars[i].TickVolume;
                shortMA /= 5;

                for (int i = currentIndex - 19; i <= currentIndex; i++)
                    longMA += bars[i].TickVolume;
                longMA /= 20;

                var oscillator = SafeDiv(shortMA - longMA, longMA) * 100;
                output.AddFeature("m5_volume_oscillator", oscillator);
            }

            // ===== VOLUME-PRICE TREND =====
            if (_volumeValues.Count >= 10 && currentIndex >= 10)
            {
                var volumeTrend = (_volumeValues[0] - _volumeValues[9]) / _volumeValues[9];
                var priceTrend = SafeDiv(
                    (double)(bar.Close - bars[currentIndex - 10].Close),
                    (double)bars[currentIndex - 10].Close);

                var correlation = volumeTrend * priceTrend > 0 ? 1.0 : -1.0;
                output.AddFeature("m5_volume_price_trend", correlation);
            }
            else
            {
                output.AddFeature("m5_volume_price_trend", 0.0);
            }
        }

        private double CalculateMFI(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period) return 50;

            double positiveFlow = 0;
            double negativeFlow = 0;

            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                var typicalPrice = (double)bars[i].TypicalPrice;
                var prevTypicalPrice = (double)bars[i - 1].TypicalPrice;
                var moneyFlow = typicalPrice * bars[i].TickVolume;

                if (typicalPrice > prevTypicalPrice)
                    positiveFlow += moneyFlow;
                else
                    negativeFlow += moneyFlow;
            }

            var moneyRatio = SafeDiv(positiveFlow, negativeFlow, 1.0);
            return 100 - (100 / (1 + moneyRatio));
        }

        private double CalculateCMF(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period) return 0;

            double sumMoneyFlow = 0;
            double sumVolume = 0;

            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                var bar = bars[i];
                var clv = SafeDiv(
                    (double)((bar.Close - bar.Low) - (bar.High - bar.Close)),
                    (double)(bar.High - bar.Low));

                sumMoneyFlow += clv * bar.TickVolume;
                sumVolume += bar.TickVolume;
            }

            return SafeDiv(sumMoneyFlow, sumVolume);
        }

        private double CalculateEMV(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period) return 0;

            var emvValues = new List<double>();

            for (int i = currentIndex - period + 2; i <= currentIndex; i++)
            {
                var bar = bars[i];
                var prevBar = bars[i - 1];

                var distance = (double)((bar.High + bar.Low) / 2 - (prevBar.High + prevBar.Low) / 2);
                var boxRatio = SafeDiv(
                    (double)bar.TickVolume / 10000,
                    (double)(bar.High - bar.Low), 1.0);

                var emv = SafeDiv(distance, boxRatio);
                emvValues.Add(emv);
            }

            return emvValues.Average();
        }

        public override void Reset()
        {
            _volumeValues.Clear();
            _obv.Clear();
        }
    }
}
