using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.M5
{
    public class M5VolatilityFeatures : BaseFeatureCalculator
    {
        public override string Name => "M5_Volatility";
        public override string Category => "Volatility";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(5);
        public override int Priority => 25;

        private readonly RollingWindow<double> _volatilityValues = new(50);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 1) return;

            var bar = bars[currentIndex];
            var close = (double)bar.Close;

            // ===== ATR =====
            if (currentIndex >= 14)
            {
                var atr14 = ATR(bars, 14, currentIndex);
                output.AddFeature("m5_atr_14", atr14 * 10000);
                output.AddFeature("m5_normalized_atr", SafeDiv(atr14, close) * 10000);

                _volatilityValues.Add(atr14);

                // ATR Expansion
                if (currentIndex >= 28)
                {
                    var atr14_prev = ATR(bars, 14, currentIndex - 14);
                    var expansion = SafeDiv(atr14 - atr14_prev, atr14_prev);
                    output.AddFeature("m5_atr_expansion", expansion);
                }
            }

            if (currentIndex >= 21)
            {
                var atr21 = ATR(bars, 21, currentIndex);
                output.AddFeature("m5_atr_21", atr21 * 10000);

                if (currentIndex >= 14)
                {
                    var atr14 = ATR(bars, 14, currentIndex);
                    output.AddFeature("m5_atr_ratio", SafeDiv(atr14, atr21));
                }
            }

            // ===== BOLLINGER BANDS =====
            if (currentIndex >= 20)
            {
                var sma20 = SMA(bars, 20, currentIndex);
                var stdDev = StdDev(bars, 20, currentIndex);

                var bbUpper = sma20 + 2 * stdDev;
                var bbLower = sma20 - 2 * stdDev;
                var bbWidth = bbUpper - bbLower;

                output.AddFeature("m5_bb_upper", bbUpper);
                output.AddFeature("m5_bb_lower", bbLower);
                output.AddFeature("m5_bb_width", SafeDiv(bbWidth, sma20) * 10000);
                output.AddFeature("m5_bb_percent_b", SafeDiv(close - bbLower, bbWidth));

                // Bollinger Band Squeeze
                if (currentIndex >= 40)
                {
                    var avgWidth = 0.0;
                    for (int i = currentIndex - 19; i <= currentIndex; i++)
                    {
                        var s = SMA(bars, 20, i);
                        var sd = StdDev(bars, 20, i);
                        avgWidth += (2 * sd) / s;
                    }
                    avgWidth /= 20;

                    var currentWidth = SafeDiv(bbWidth, sma20);
                    var squeeze = currentWidth < avgWidth * 0.8 ? 1.0 : 0.0;
                    output.AddFeature("m5_bb_squeeze", squeeze);
                }
            }

            // ===== HISTORICAL VOLATILITY =====
            if (currentIndex >= 10)
            {
                var histVol10 = CalculateHistoricalVolatility(bars, 10, currentIndex);
                output.AddFeature("m5_hist_vol_10", histVol10);
            }

            if (currentIndex >= 20)
            {
                var histVol20 = CalculateHistoricalVolatility(bars, 20, currentIndex);
                output.AddFeature("m5_hist_vol_20", histVol20);

                if (currentIndex >= 10)
                {
                    var histVol10 = CalculateHistoricalVolatility(bars, 10, currentIndex);
                    output.AddFeature("m5_vol_ratio", SafeDiv(histVol10, histVol20));
                }
            }

            // ===== RANGE-BASED =====
            var trueRange = TrueRange(bars, currentIndex);
            output.AddFeature("m5_true_range", trueRange * 10000);

            if (currentIndex >= 10)
            {
                var avgRange = 0.0;
                for (int i = currentIndex - 9; i <= currentIndex; i++)
                {
                    avgRange += TrueRange(bars, i);
                }
                avgRange /= 10;

                var expansion = trueRange > avgRange * 1.5 ? 1.0 : 0.0;
                var contraction = trueRange < avgRange * 0.5 ? 1.0 : 0.0;

                output.AddFeature("m5_range_expansion", expansion);
                output.AddFeature("m5_range_contraction", contraction);
            }

            // ===== CHAIKIN VOLATILITY =====
            if (currentIndex >= 20)
            {
                var chaikinVol = CalculateChaikinVolatility(bars, 10, currentIndex);
                output.AddFeature("m5_chaikin_volatility", chaikinVol);
            }

            // ===== VOLATILITY TREND =====
            if (_volatilityValues.Count >= 10)
            {
                var recent = _volatilityValues.GetValues().Take(10).ToArray();
                var older = _volatilityValues.GetValues().Skip(10).Take(10).ToArray();

                if (older.Length == 10)
                {
                    var recentAvg = recent.Average();
                    var olderAvg = older.Average();
                    var trend = SafeDiv(recentAvg - olderAvg, olderAvg);
                    output.AddFeature("m5_volatility_trend", trend);
                }
                else
                {
                    output.AddFeature("m5_volatility_trend", 0.0);
                }
            }
            else
            {
                output.AddFeature("m5_volatility_trend", 0.0);
            }
        }

        private double CalculateHistoricalVolatility(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period) return 0;

            var returns = new List<double>();
            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                var ret = Math.Log((double)bars[i].Close / (double)bars[i - 1].Close);
                returns.Add(ret);
            }

            var mean = returns.Average();
            var variance = returns.Select(r => Math.Pow(r - mean, 2)).Average();

            return Math.Sqrt(variance * 252) * 100; // Annualized
        }

        private double CalculateChaikinVolatility(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period * 2) return 0;

            var hlDiff = new List<double>();
            for (int i = currentIndex - period * 2 + 1; i <= currentIndex; i++)
            {
                hlDiff.Add((double)(bars[i].High - bars[i].Low));
            }

            var ema1 = hlDiff.Skip(period).Average();
            var ema2 = hlDiff.Take(period).Average();

            return SafeDiv(ema1 - ema2, ema2) * 100;
        }

        public override void Reset()
        {
            _volatilityValues.Clear();
        }
    }
}
