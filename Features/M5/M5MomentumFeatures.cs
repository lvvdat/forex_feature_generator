using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.M5
{
    public class M5MomentumFeatures : BaseFeatureCalculator
    {
        public override string Name => "M5_Momentum";
        public override string Category => "Momentum";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(5);
        public override int Priority => 22;

        private readonly RollingWindow<double> _rsiValues = new(50);
        private readonly RollingWindow<double> _momentumValues = new(50);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 1) return;

            var bar = bars[currentIndex];
            var close = (double)bar.Close;

            // ===== RSI =====
            if (currentIndex >= 14)
            {
                var rsi14 = CalculateRSI(bars, 14, currentIndex);
                output.AddFeature("fg3_rsi_14", rsi14);
                output.AddFeature("fg3_rsi_oversold", rsi14 < 30 ? 1.0 : 0.0);
                output.AddFeature("fg3_rsi_overbought", rsi14 > 70 ? 1.0 : 0.0);

                _rsiValues.Add(rsi14);

                // RSI Divergence
                if (currentIndex >= 20)
                {
                    var priceTrend = CalculateTrend(bars, currentIndex, 5, true);
                    var rsiTrend = CalculateRSITrend(bars, currentIndex, 5);
                    var divergence = (priceTrend > 0 && rsiTrend < 0) ? -1.0 : (priceTrend < 0 && rsiTrend > 0) ? 1.0 : 0.0;
                    output.AddFeature("fg3_rsi_divergence", divergence);
                }
            }

            if (currentIndex >= 21)
            {
                output.AddFeature("fg3_rsi_21", CalculateRSI(bars, 21, currentIndex));
            }

            // ===== STOCHASTIC =====
            if (currentIndex >= 14)
            {
                var (stochK, stochD) = CalculateStochastic(bars, 14, 3, currentIndex);
                output.AddFeature("fg3_stoch_k_14", stochK);
                output.AddFeature("fg3_stoch_d_14", stochD);

                // Stochastic Divergence
                if (currentIndex >= 20)
                {
                    var priceTrend = CalculateTrend(bars, currentIndex, 5, true);
                    var stochTrend = CalculateStochTrend(bars, currentIndex, 5);
                    var divergence = (priceTrend > 0 && stochTrend < 0) ? -1.0 : (priceTrend < 0 && stochTrend > 0) ? 1.0 : 0.0;
                    output.AddFeature("fg3_stoch_divergence", divergence);
                }
            }

            // ===== MACD =====
            if (currentIndex >= 26)
            {
                var ema12 = EMA(bars, 12, currentIndex);
                var ema26 = EMA(bars, 26, currentIndex);
                var macdLine = ema12 - ema26;

                output.AddFeature("fg3_macd_line", macdLine * 10000);

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

                    output.AddFeature("fg3_macd_signal", macdSignal * 10000);
                    output.AddFeature("fg3_macd_histogram", macdHist * 10000);

                    // MACD Cross
                    if (currentIndex >= 36)
                    {
                        var prevMacdLine = EMA(bars, 12, currentIndex - 1) - EMA(bars, 26, currentIndex - 1);
                        var cross = 0.0;
                        if (macdLine > macdSignal && prevMacdLine <= macdSignal) cross = 1.0;
                        else if (macdLine < macdSignal && prevMacdLine >= macdSignal) cross = -1.0;
                        output.AddFeature("fg3_macd_cross", cross);
                    }
                }
            }

            // ===== MOMENTUM =====
            if (currentIndex >= 10)
            {
                var momentum = close - (double)bars[currentIndex - 10].Close;
                output.AddFeature("fg3_momentum_10", momentum * 10000);
                _momentumValues.Add(momentum);

                // Rate of Change
                var roc = SafeDiv(momentum, (double)bars[currentIndex - 10].Close) * 100;
                output.AddFeature("fg3_roc_10", roc);

                // Momentum Acceleration
                if (_momentumValues.Count >= 2)
                {
                    var acceleration = _momentumValues[0] - _momentumValues[1];
                    output.AddFeature("fg3_momentum_acceleration", acceleration * 10000);
                }
                else
                {
                    output.AddFeature("fg3_momentum_acceleration", 0.0);
                }

                // Momentum Quality (consistency)
                if (_momentumValues.Count >= 5)
                {
                    var recent = _momentumValues.GetValues().Take(5).ToArray();
                    var positiveCount = recent.Count(m => m > 0);
                    var quality = (positiveCount / 5.0) * 2 - 1; // Range: -1 to 1
                    output.AddFeature("fg3_momentum_quality", quality);
                }
                else
                {
                    output.AddFeature("fg3_momentum_quality", 0.0);
                }
            }

            // ===== WILLIAMS %R =====
            if (currentIndex >= 14)
            {
                var williamsR = CalculateWilliamsR(bars, 14, currentIndex);
                output.AddFeature("fg3_williams_r", williamsR);
            }

            // ===== ULTIMATE OSCILLATOR =====
            if (currentIndex >= 28)
            {
                var uo = CalculateUltimateOscillator(bars, currentIndex);
                output.AddFeature("fg3_ultimate_oscillator", uo);
            }

            // ===== MOMENTUM DIVERGENCE =====
            if (currentIndex >= 20)
            {
                var priceTrend = CalculateTrend(bars, currentIndex, 10, true);
                var momentumTrend = CalculateTrend(bars, currentIndex, 10, false);
                var divergence = (priceTrend > 0 && momentumTrend < 0) ? -1.0 : (priceTrend < 0 && momentumTrend > 0) ? 1.0 : 0.0;
                output.AddFeature("fg3_momentum_divergence", divergence);
            }
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


        private double CalculateWilliamsR(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period) return -50;

            double highest = double.MinValue;
            double lowest = double.MaxValue;

            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                highest = Math.Max(highest, (double)bars[i].High);
                lowest = Math.Min(lowest, (double)bars[i].Low);
            }

            var close = (double)bars[currentIndex].Close;
            return SafeDiv(highest - close, highest - lowest) * -100;
        }

        private double CalculateUltimateOscillator(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Periods: 7, 14, 28
            double bp7 = 0, bp14 = 0, bp28 = 0;
            double tr7 = 0, tr14 = 0, tr28 = 0;

            for (int i = 1; i <= 28 && currentIndex - i >= 0; i++)
            {
                var curr = bars[currentIndex - i + 1];
                var prev = bars[currentIndex - i];

                var bp = (double)(curr.Close - Math.Min(curr.Low, prev.Close));
                var tr = TrueRange(bars, currentIndex - i + 1);

                if (i <= 7) { bp7 += bp; tr7 += tr; }
                if (i <= 14) { bp14 += bp; tr14 += tr; }
                bp28 += bp; tr28 += tr;
            }

            var avg7 = SafeDiv(bp7, tr7);
            var avg14 = SafeDiv(bp14, tr14);
            var avg28 = SafeDiv(bp28, tr28);

            return ((avg7 * 4) + (avg14 * 2) + avg28) / 7 * 100;
        }

        private double CalculateTrend(IReadOnlyList<OhlcBar> bars, int currentIndex, int period, bool usePrice)
        {
            if (currentIndex < period) return 0;

            double sum = 0;
            for (int i = 1; i < period; i++)
            {
                if (usePrice)
                {
                    sum += (double)(bars[currentIndex - i + 1].Close - bars[currentIndex - i].Close);
                }
                else
                {
                    var momentum1 = (double)(bars[currentIndex - i + 1].Close - bars[currentIndex - i - 9].Close);
                    var momentum2 = (double)(bars[currentIndex - i].Close - bars[currentIndex - i - 10].Close);
                    sum += momentum1 - momentum2;
                }
            }
            return sum;
        }

        private double CalculateRSITrend(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            if (currentIndex < period + 14) return 0;

            double sum = 0;
            for (int i = 1; i < period; i++)
            {
                var rsi1 = CalculateRSI(bars, 14, currentIndex - i + 1);
                var rsi2 = CalculateRSI(bars, 14, currentIndex - i);
                sum += rsi1 - rsi2;
            }
            return sum;
        }

        private double CalculateStochTrend(IReadOnlyList<OhlcBar> bars, int currentIndex, int period)
        {
            if (currentIndex < period + 14) return 0;

            double sum = 0;
            for (int i = 1; i < period; i++)
            {
                var (k1, _) = CalculateStochastic(bars, 14, 3, currentIndex - i + 1);
                var (k2, _) = CalculateStochastic(bars, 14, 3, currentIndex - i);
                sum += k1 - k2;
            }
            return sum;
        }

        public override void Reset()
        {
            _rsiValues.Clear();
            _momentumValues.Clear();
        }
    }

}
