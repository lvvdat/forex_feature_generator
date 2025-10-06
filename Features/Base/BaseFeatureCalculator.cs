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

        protected double[] GetCloseArray(IReadOnlyList<OhlcBar> bars, int start, int count)
        {
            var result = new double[count];
            for (int i = 0; i < count; i++)
            {
                result[i] = (double)bars[start + i].Close;
            }
            return result;
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
    }
}
