using ForexFeatureGenerator.Core.Models;

namespace ForexFeatureGenerator.Features.Core
{
    /// <summary>
    /// Base feature calculator optimized for 3-class prediction (-1, 0, 1)
    /// Provides common transformations and utilities for directional classification
    /// </summary>
    public abstract class BaseFeatureCalculator
    {
        public abstract string Name { get; }
        public abstract string Category { get; }
        public abstract TimeSpan Timeframe { get; }
        public abstract int Priority { get; }
        public bool IsEnabled { get; set; } = true;

        // ===== CORE CALCULATION METHOD =====
        public abstract void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex);
        public virtual void Reset() { }

        // ===== TRANSFORMATION UTILITIES FOR 3-CLASS PREDICTION =====

        /// <summary>
        /// Converts continuous value to z-score for better discrimination
        /// Z-scores help identify extreme moves that trigger directional labels
        /// </summary>
        protected double CalculateZScore(double value, double mean, double stdDev)
        {
            if (stdDev < 1e-10) return 0;
            return (value - mean) / stdDev;
        }

        /// <summary>
        /// Converts value to percentile rank (0-100) for relative strength
        /// Helps identify overbought/oversold conditions
        /// </summary>
        protected double CalculatePercentileRank(double value, List<double> historicalValues)
        {
            if (historicalValues.Count == 0) return 50;
            var count = historicalValues.Count(v => v < value);
            return (double)count / historicalValues.Count * 100;
        }

        /// <summary>
        /// Creates directional signal from indicator value and thresholds
        /// Returns: 1 for bullish, -1 for bearish, 0 for neutral
        /// </summary>
        protected double CreateDirectionalSignal(double value, double bullishThreshold, double bearishThreshold)
        {
            if (value > bullishThreshold) return 1.0;
            if (value < bearishThreshold) return -1.0;
            return 0.0;
        }

        /// <summary>
        /// Calculates momentum quality - how consistent the move is
        /// Higher quality = more likely to continue in direction
        /// </summary>
        protected double CalculateMomentumQuality(IReadOnlyList<double> values)
        {
            if (values.Count < 2) return 0;

            int consistentMoves = 0;
            for (int i = 1; i < values.Count; i++)
            {
                if (Math.Sign(values[i] - values[i - 1]) == Math.Sign(values[0] - values[1]))
                    consistentMoves++;
            }

            return (double)consistentMoves / (values.Count - 1);
        }

        /// <summary>
        /// Detects divergence between price and indicator
        /// Critical for identifying potential reversals
        /// </summary>
        protected double CalculateDivergence(double[] prices, double[] indicator, int lookback = 10)
        {
            if (prices.Length < lookback || indicator.Length < lookback) return 0;

            // Calculate slopes
            var priceSlope = CalculateSlope(prices.TakeLast(lookback).ToArray());
            var indicatorSlope = CalculateSlope(indicator.TakeLast(lookback).ToArray());

            // Bullish divergence: price down, indicator up
            if (priceSlope < -0.0001 && indicatorSlope > 0.0001) return 1.0;

            // Bearish divergence: price up, indicator down
            if (priceSlope > 0.0001 && indicatorSlope < -0.0001) return -1.0;

            return 0.0;
        }

        /// <summary>
        /// Calculates normalized distance from support/resistance
        /// Helps predict bounces or breakouts
        /// </summary>
        protected double CalculateNormalizedDistance(double price, double level, double atr)
        {
            if (atr < 1e-10) return 0;
            return (price - level) / atr;
        }

        /// <summary>
        /// Creates composite signal from multiple indicators
        /// Weighted voting for more robust predictions
        /// </summary>
        protected double CreateCompositeSignal(params (double signal, double weight)[] signals)
        {
            double weightedSum = 0;
            double totalWeight = 0;

            foreach (var (signal, weight) in signals)
            {
                weightedSum += signal * weight;
                totalWeight += weight;
            }

            if (totalWeight < 1e-10) return 0;

            var composite = weightedSum / totalWeight;

            // Apply thresholds for clear signals
            if (composite > 0.5) return 1.0;
            if (composite < -0.5) return -1.0;
            return 0.0;
        }

        /// <summary>
        /// Calculates market regime for context-aware features
        /// Different regimes require different feature interpretations
        /// </summary>
        protected int DetectMarketRegime(double volatility, double trendStrength, double volume)
        {
            // 0: Range, 1: Trending, 2: Volatile/News
            if (volatility > 1.5) return 2;  // High volatility regime
            if (trendStrength > 0.7) return 1;  // Trending regime
            return 0;  // Range-bound regime
        }

        /// <summary>
        /// Safe division preventing NaN/Infinity
        /// </summary>
        protected double SafeDiv(double numerator, double denominator, double defaultValue = 0)
        {
            if (Math.Abs(denominator) < 1e-10) return defaultValue;
            var result = numerator / denominator;
            return double.IsNaN(result) || double.IsInfinity(result) ? defaultValue : result;
        }

        /// <summary>
        /// Calculates linear regression slope for trend detection
        /// </summary>
        protected double CalculateSlope(double[] values)
        {
            if (values.Length < 2) return 0;

            var n = values.Length;
            var xValues = Enumerable.Range(0, n).Select(i => (double)i).ToArray();

            var sumX = xValues.Sum();
            var sumY = values.Sum();
            var sumXY = xValues.Zip(values, (x, y) => x * y).Sum();
            var sumX2 = xValues.Sum(x => x * x);

            return SafeDiv(n * sumXY - sumX * sumY, n * sumX2 - sumX * sumX);
        }

        /// <summary>
        /// Calculates adaptive moving average for dynamic markets
        /// </summary>
        protected double CalculateAdaptiveMA(IReadOnlyList<OhlcBar> bars, int currentIndex,
            int fastPeriod = 5, int slowPeriod = 20)
        {
            if (currentIndex < slowPeriod) return (double)bars[currentIndex].Close;

            // Calculate efficiency ratio
            var direction = Math.Abs((double)(bars[currentIndex].Close - bars[currentIndex - slowPeriod].Close));
            var volatility = 0.0;

            for (int i = currentIndex - slowPeriod + 1; i <= currentIndex; i++)
            {
                volatility += Math.Abs((double)(bars[i].Close - bars[i - 1].Close));
            }

            var efficiencyRatio = SafeDiv(direction, volatility, 0.5);

            // Calculate smoothing constant
            var fastSC = 2.0 / (fastPeriod + 1);
            var slowSC = 2.0 / (slowPeriod + 1);
            var sc = Math.Pow(efficiencyRatio * (fastSC - slowSC) + slowSC, 2);

            // Calculate AMA
            double ama = (double)bars[currentIndex - slowPeriod].Close;
            for (int i = currentIndex - slowPeriod + 1; i <= currentIndex; i++)
            {
                ama = ama + sc * ((double)bars[i].Close - ama);
            }

            return ama;
        }

        /// <summary>
        /// Normalizes feature to [-1, 1] range for better model training
        /// </summary>
        protected double NormalizeToRange(double value, double min, double max)
        {
            if (max - min < 1e-10) return 0;
            return 2 * (value - min) / (max - min) - 1;
        }

        /// <summary>
        /// Applies sigmoid transformation for probability-like features
        /// </summary>
        protected double Sigmoid(double x, double steepness = 1.0)
        {
            return 2.0 / (1.0 + Math.Exp(-steepness * x)) - 1.0;
        }
    }
}