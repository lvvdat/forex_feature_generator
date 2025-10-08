using ForexFeatureGenerator.Core.Models;

namespace ForexFeatureGenerator.Features.Base
{
    public interface IFeatureCalculator
    {
        string Name { get; }
        string Category { get; }
        TimeSpan Timeframe { get; }
        bool IsEnabled { get; set; }
        int Priority { get; }

        void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex);
        void Reset();
    }

    public abstract class BaseFeatureCalculator : IFeatureCalculator
    {
        public abstract string Name { get; }
        public abstract string Category { get; }
        public abstract TimeSpan Timeframe { get; }
        public bool IsEnabled { get; set; } = true;
        public virtual int Priority => 100;

        public abstract void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex);
        public virtual void Reset() { }

        protected double SafeDiv(double numerator, double denominator, double defaultValue = 0.0)
        {
            if (Math.Abs(denominator) < 1e-10) return defaultValue;
            var result = numerator / denominator;
            return double.IsNaN(result) || double.IsInfinity(result) ? defaultValue : result;
        }

        // FIXED: Proper EMA calculation with SMA initialization
        protected double EMA(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period - 1 || period <= 0) return 0;

            var multiplier = 2.0 / (period + 1);

            // Initialize with SMA of first 'period' bars
            double ema = 0;
            int startIdx = currentIndex - period + 1;
            for (int i = 0; i < period; i++)
            {
                ema += (double)bars[startIdx + i].Close;
            }
            ema /= period;

            // Calculate EMA for remaining bars
            for (int i = currentIndex - period + 1 + period; i <= currentIndex; i++)
            {
                ema = ((double)bars[i].Close - ema) * multiplier + ema;
            }

            return ema;
        }

        protected double SMA(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period - 1) return 0;

            double sum = 0;
            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                sum += (double)bars[i].Close;
            }
            return sum / period;
        }

        protected double StdDev(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period - 1) return 0;

            var mean = SMA(bars, period, currentIndex);
            double sumSquares = 0;

            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                var diff = (double)bars[i].Close - mean;
                sumSquares += diff * diff;
            }

            return Math.Sqrt(sumSquares / period);
        }

        protected double ATR(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period) return 0;

            double sum = 0;
            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                var tr = TrueRange(bars, i);
                sum += tr;
            }

            return sum / period;
        }

        protected double TrueRange(IReadOnlyList<OhlcBar> bars, int index)
        {
            if (index < 1) return (double)(bars[index].High - bars[index].Low);

            var high = (double)bars[index].High;
            var low = (double)bars[index].Low;
            var prevClose = (double)bars[index - 1].Close;

            return Math.Max(
                high - low,
                Math.Max(
                    Math.Abs(high - prevClose),
                    Math.Abs(low - prevClose)
                )
            );
        }

        // Stochastic Oscillator: %K (with smoothing) and %D (3-SMA of %K)
        protected (double k, double d) CalculateStochastic(
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
    }
}
