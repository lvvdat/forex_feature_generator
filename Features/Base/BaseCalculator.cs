using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Core.Infrastructure;
using System;
using System.Collections.Generic;
using System.Linq;

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

    public abstract class BaseCalculator : IFeatureCalculator
    {
        public abstract string Name { get; }
        public abstract string Category { get; }
        public abstract TimeSpan Timeframe { get; }
        public bool IsEnabled { get; set; } = true;
        public virtual int Priority => 100;

        // Rolling windows for adaptive normalization
        protected readonly RollingWindow<double> _adaptiveWindow = new(100);

        public abstract void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex);
        public virtual void Reset()
        {
            _adaptiveWindow.Clear();
        }

        // === Core Calculation Helpers ===

        protected double SafeDiv(double numerator, double denominator, double defaultValue = 0.0)
        {
            if (Math.Abs(denominator) < 1e-10) return defaultValue;
            var result = numerator / denominator;
            return double.IsNaN(result) || double.IsInfinity(result) ? defaultValue : result;
        }

        // Adaptive Z-score normalization
        protected double AdaptiveZScore(double value, double windowMean, double windowStd)
        {
            if (windowStd < 1e-10) return 0;
            var zScore = (value - windowMean) / windowStd;
            // Clip to [-3, 3] range for stability
            return Math.Max(-3, Math.Min(3, zScore));
        }

        // Rank transformation (percentile in rolling window)
        protected double RankTransform(double value, List<double> window)
        {
            if (window.Count == 0) return 0.5;
            int rank = window.Count(v => v < value);
            return (double)rank / window.Count;
        }

        // Sigmoid transformation for bounded output
        protected double Sigmoid(double x, double scale = 1.0)
        {
            return 1.0 / (1.0 + Math.Exp(-x * scale));
        }

        // === Technical Indicators ===

        protected double EMA(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period - 1 || period <= 0) return 0;

            var multiplier = 2.0 / (period + 1);

            // Initialize with SMA
            double ema = 0;
            int startIdx = currentIndex - period + 1;
            for (int i = 0; i < period; i++)
            {
                ema += (double)bars[startIdx + i].Close;
            }
            ema /= period;

            // Apply EMA calculation for any remaining bars
            for (int i = startIdx + period; i <= currentIndex; i++)
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
                sum += TrueRange(bars, i);
            }

            return sum / period;
        }

        protected double TrueRange(IReadOnlyList<OhlcBar> bars, int index)
        {
            if (index < 1) return (double)(bars[index].High - bars[index].Low);

            var high = (double)bars[index].High;
            var low = (double)bars[index].Low;
            var prevClose = (double)bars[index - 1].Close;

            return Math.Max(high - low, Math.Max(Math.Abs(high - prevClose), Math.Abs(low - prevClose)));
        }

        protected double RSI(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
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

            if (losses < 1e-10) return 100;

            var rs = gains / losses;
            return 100 - (100 / (1 + rs));
        }

        // Linear regression slope
        protected double CalculateSlope(double[] values)
        {
            int n = values.Length;
            if (n < 2) return 0;

            double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
            for (int i = 0; i < n; i++)
            {
                sumX += i;
                sumY += values[i];
                sumXY += i * values[i];
                sumX2 += i * i;
            }

            return SafeDiv(n * sumXY - sumX * sumY, n * sumX2 - sumX * sumX);
        }
    }
}