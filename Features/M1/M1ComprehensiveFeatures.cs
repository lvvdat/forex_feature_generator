using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;

namespace ForexFeatureGenerator.Features.M1
{
    public class M1ComprehensiveFeatures : BaseFeatureCalculator
    {
        public override string Name => "M1_Comprehensive";
        public override string Category => "Comprehensive";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(1);
        public override int Priority => 3;

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 1) return;

            var bar = bars[currentIndex];

            // ===== VOLUME-BASED FEATURES =====
            output.AddFeature("fg1_up_volume_pct", (double)bar.UpVolume / bar.TickVolume * 100);
            output.AddFeature("fg1_down_volume_pct", (double)bar.DownVolume / bar.TickVolume * 100);

            var volume_imbalance = (double)(bar.UpVolume - bar.DownVolume);
            output.AddFeature("fg1_volume_imbalance", volume_imbalance / bar.TickVolume);

            // ===== PRICE-BASED FEATURES =====
            var range = (double)(bar.High - bar.Low);
            var close = (double)bar.Close;

            if (range > 0)
            {
                var upMove = (double)(bar.High - bar.Open);
                var downMove = (double)(bar.Open - bar.Low);

                output.AddFeature("fg1_up_price_pct", SafeDiv(upMove, range) * 100);
                output.AddFeature("fg1_down_price_pct", SafeDiv(downMove, range) * 100);
            }
            else
            {
                output.AddFeature("fg1_up_price_pct", 0);
                output.AddFeature("fg1_down_price_pct", 0);
            }

            output.AddFeature("fg1_range_pct", SafeDiv(range, close) * 100);

            // ===== ENTROPY =====
            if (currentIndex >= 20)
            {
                var returns = new List<double>();
                for (int i = currentIndex - 19; i <= currentIndex; i++)
                {
                    if (i > 0)
                    {
                        var ret = Math.Log((double)bars[i].Close / (double)bars[i - 1].Close);
                        returns.Add(ret);
                    }
                }

                var entropy = CalculateEntropy(returns);
                output.AddFeature("fg1_entropy_20", entropy);
            }

            // ===== RSI =====
            if (currentIndex >= 9)
            {
                output.AddFeature("fg1_rsi_9", CalculateRSI(bars, 9, currentIndex));
            }
            if (currentIndex >= 14)
            {
                output.AddFeature("fg1_rsi_14", CalculateRSI(bars, 14, currentIndex));
            }

            // ===== ATR =====
            if (currentIndex >= 10)
            {
                var atr10 = ATR(bars, 10, currentIndex);
                output.AddFeature("fg1_atr_10", atr10);
                output.AddFeature("fg1_atr_pct", SafeDiv(atr10, close) * 100);
            }
            if (currentIndex >= 14)
            {
                output.AddFeature("fg1_atr_14", ATR(bars, 14, currentIndex));
            }

            // ===== STOCHASTIC =====
            if (currentIndex >= 14)
            {
                var (stochK, stochD) = CalculateStochastic(bars, 14, 3, currentIndex);
                output.AddFeature("fg1_stoch_k", stochK);
                output.AddFeature("fg1_stoch_d", stochD);
            }

            // ===== BOLLINGER BANDS =====
            if (currentIndex >= 20)
            {
                var sma = SMA(bars, 20, currentIndex);
                var std = StdDev(bars, 20, currentIndex);
                var bbUpper = sma + 2 * std;
                var bbLower = sma - 2 * std;

                output.AddFeature("fg1_bb_upper", bbUpper);
                output.AddFeature("fg1_bb_lower", bbLower);
                output.AddFeature("fg1_bb_width", SafeDiv(bbUpper - bbLower, sma) * 100);
                output.AddFeature("fg1_bb_pct", SafeDiv(close - bbLower, bbUpper - bbLower) * 100);
            }

            // ===== MACD =====
            if (currentIndex >= 26)
            {
                var ema12 = EMA(bars, 12, currentIndex);
                var ema26 = EMA(bars, 26, currentIndex);
                var macdLine = ema12 - ema26;

                output.AddFeature("fg1_macd_line", macdLine * 10000);

                if (currentIndex >= 35)
                {
                    // Calculate signal line (9-period EMA of MACD)
                    var macdValues = new List<double>();
                    for (int i = currentIndex - 8; i <= currentIndex; i++)
                    {
                        var e12 = EMA(bars, 12, i);
                        var e26 = EMA(bars, 26, i);
                        macdValues.Add(e12 - e26);
                    }
                    var macdSignal = macdValues.Average(); // Simplified
                    var macdHist = macdLine - macdSignal;

                    output.AddFeature("fg1_macd_signal", macdSignal * 10000);
                    output.AddFeature("fg1_macd_histogram", macdHist * 10000);

                    // MACD Cross
                    if (currentIndex >= 36)
                    {
                        var prevMacdLine = EMA(bars, 12, currentIndex - 1) - EMA(bars, 26, currentIndex - 1);
                        var cross = 0.0;
                        if (macdLine > macdSignal && prevMacdLine <= macdSignal) cross = 1.0;
                        else if (macdLine < macdSignal && prevMacdLine >= macdSignal) cross = -1.0;
                        output.AddFeature("fg1_macd_cross", cross);
                    }
                }
            }

            // ===== AROON =====
            if (currentIndex >= 25)
            {
                var (aroonUp, aroonDown) = CalculateAroon(bars, 25, currentIndex);
                output.AddFeature("fg1_aroon_up", aroonUp);
                output.AddFeature("fg1_aroon_down", aroonDown);
                output.AddFeature("fg1_aroon_osc", aroonUp - aroonDown);
            }

            // ===== CCI =====
            if (currentIndex >= 20)
            {
                var cci = CalculateCCI(bars, 20, currentIndex);
                output.AddFeature("fg1_cci_20", cci);
            }
        }

        private double CalculateEntropy(List<double> returns)
        {
            if (returns.Count == 0) return 0;

            // Discretize returns into bins
            var bins = 10;
            var min = returns.Min();
            var max = returns.Max();
            var binWidth = (max - min) / bins;

            if (binWidth < 1e-10) return 0;

            var counts = new int[bins];
            foreach (var ret in returns)
            {
                var bin = (int)((ret - min) / binWidth);
                if (bin >= bins) bin = bins - 1;
                if (bin < 0) bin = 0;
                counts[bin]++;
            }

            // Calculate Shannon entropy
            double entropy = 0;
            foreach (var count in counts)
            {
                if (count > 0)
                {
                    var p = (double)count / returns.Count;
                    entropy -= p * Math.Log(p, 2);
                }
            }

            return entropy;
        }

        private double CalculateRSI(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period) return 50;

            double gains = 0;
            double losses = 0;

            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                var change = (double)(bars[i].Close - bars[i - 1].Close);
                if (change > 0)
                    gains += change;
                else
                    losses += Math.Abs(change);
            }

            var avgGain = gains / period;
            var avgLoss = losses / period;

            if (avgLoss < 1e-10) return 100;

            var rs = avgGain / avgLoss;
            return 100 - (100 / (1 + rs));
        }

        // Stochastic Oscillator: %K (with smoothing) and %D (3-SMA of %K)
        private (double k, double d) CalculateStochastic(
            IReadOnlyList<OhlcBar> bars,
            int period,
            int smoothK,
            int currentIndex)
        {
            // --- Validate inputs ---
            if (bars == null || bars.Count == 0) return (50, 50);
            if (period <= 0) period = 14;          // sensible default
            if (smoothK <= 0) smoothK = 1;         // 1 => no smoothing of %K
            const int dPeriod = 3;                  // standard %D = 3-SMA of %K

            // Earliest index needed to compute %D of smoothed %K at currentIndex:
            // raw %K requires (period - 1) lookback
            // smoothed %K requires (smoothK - 1) more
            // %D requires (dPeriod - 1) more
            int minIndexNeeded = (period - 1) + (smoothK - 1) + (dPeriod - 1);
            if (currentIndex < minIndexNeeded) return (50, 50);
            if (currentIndex >= bars.Count) currentIndex = bars.Count - 1;

            // Helper: compute raw %K at a given index (no smoothing)
            double RawKAt(int idx)
            {
                int start = idx - period + 1;
                double highestHigh = double.MinValue;
                double lowestLow = double.MaxValue;

                for (int i = start; i <= idx; i++)
                {
                    double hi = (double)bars[i].High;
                    double lo = (double)bars[i].Low;
                    if (hi > highestHigh) highestHigh = hi;
                    if (lo < lowestLow) lowestLow = lo;
                }

                double close = (double)bars[idx].Close;
                double range = highestHigh - lowestLow;

                if (range <= 0.0) return 50.0; // flat range -> neutral
                return ((close - lowestLow) / range) * 100.0;
            }

            // Helper: simple moving average over f(idx) for N points ending at idx
            double SMA(Func<int, double> f, int idx, int N)
            {
                double sum = 0.0;
                for (int i = idx - N + 1; i <= idx; i++)
                    sum += f(i);
                return sum / N;
            }

            // Smoothed %K at an index: SMA of raw %K over 'smoothK' bars
            double SmoothedKAt(int idx)
            {
                if (smoothK == 1) return RawKAt(idx);
                return SMA(RawKAt, idx, smoothK);
            }

            // Current %K and %D
            double k = SmoothedKAt(currentIndex);
            double d = SMA(SmoothedKAt, currentIndex, dPeriod);

            // Clamp to [0,100] just in case of numerical quirks
            k = Math.Min(100.0, Math.Max(0.0, k));
            d = Math.Min(100.0, Math.Max(0.0, d));

            return (k, d);
        }

        private (double up, double down) CalculateAroon(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period) return (50, 50);

            int daysSinceHigh = 0;
            int daysSinceLow = 0;
            double highestHigh = double.MinValue;
            double lowestLow = double.MaxValue;

            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                var high = (double)bars[i].High;
                var low = (double)bars[i].Low;

                if (high >= highestHigh)
                {
                    highestHigh = high;
                    daysSinceHigh = currentIndex - i;
                }

                if (low <= lowestLow)
                {
                    lowestLow = low;
                    daysSinceLow = currentIndex - i;
                }
            }

            var aroonUp = ((double)(period - daysSinceHigh) / period) * 100;
            var aroonDown = ((double)(period - daysSinceLow) / period) * 100;

            return (aroonUp, aroonDown);
        }

        private double CalculateCCI(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period) return 0;

            // Calculate typical prices
            var typicalPrices = new double[period];
            for (int i = 0; i < period; i++)
            {
                var idx = currentIndex - period + 1 + i;
                typicalPrices[i] = (double)bars[idx].TypicalPrice;
            }

            var sma = typicalPrices.Average();
            var meanDeviation = typicalPrices.Select(tp => Math.Abs(tp - sma)).Average();

            var currentTP = (double)bars[currentIndex].TypicalPrice;

            if (meanDeviation < 1e-10) return 0;

            return (currentTP - sma) / (0.015 * meanDeviation);
        }

        public override void Reset()
        {
        }
    }
}