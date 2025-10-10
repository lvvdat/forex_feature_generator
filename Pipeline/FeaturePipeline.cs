using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Core;
using ForexFeatureGenerator.Core.Infrastructure;
using ForexFeatureGenerator.Pipeline;
using ForexFeatureGenerator.Features.Advanced;

namespace ForexFeatureGenerator.Features.Pipeline
{
    /// <summary>
    /// Enhanced feature pipeline optimized for 3-class label prediction
    /// Orchestrates all feature calculators and provides feature management
    /// </summary>
    public class FeaturePipeline
    {
        private readonly Dictionary<TimeSpan, IBarAggregator> _aggregators = new();
        private readonly List<BaseFeatureCalculator> _calculators = new();
        private readonly FeatureConfiguration _config;

        public FeaturePipeline(FeatureConfiguration? config = null)
        {
            _config = config ?? FeatureConfiguration.CreateOptimized3Class();
            InitializeAggregators();
            InitializeCalculators();
        }

        /// <summary>
        /// Initialize timeframe aggregators
        /// </summary>
        private void InitializeAggregators()
        {
            // Register multiple timeframes for comprehensive analysis
            RegisterAggregator(TimeSpan.FromMinutes(1), 1000);  // M1 - Microstructure
            RegisterAggregator(TimeSpan.FromMinutes(5), 500);   // M5 - Short-term
            // RegisterAggregator(TimeSpan.FromMinutes(15), 300);  // M15 - Medium-term
            // RegisterAggregator(TimeSpan.FromMinutes(30), 200);  // M30 - Medium-term
            // RegisterAggregator(TimeSpan.FromMinutes(60), 100);  // H1 - Long-term context
        }

        /// <summary>
        /// Initialize all feature calculators in priority order
        /// </summary>
        private void InitializeCalculators()
        {
            // Core directional features (highest priority)
            RegisterCalculator(new DirectionalFeatures());

            // Context and regime features
            RegisterCalculator(new MarketRegimeContextFeatures());

            // Microstructure and order flow
            RegisterCalculator(new MicrostructureOrderFlowFeatures());

            // Technical indicators
            RegisterCalculator(new TechnicalIndicatorFeatures());

            RegisterCalculator(new MachineLearningFeatures());
            RegisterCalculator(new DeepLearningFeatures());
            RegisterCalculator(new PositionFeatures());
        }

        public void RegisterAggregator(TimeSpan timeframe, int historySize)
        {
            _aggregators[timeframe] = new BarAggregator(timeframe, historySize);
        }

        public void RegisterCalculator(BaseFeatureCalculator calculator)
        {
            calculator.IsEnabled = _config.IsFeatureEnabled(calculator.Name);
            _calculators.Add(calculator);
        }

        /// <summary>
        /// Gets all feature names ordered by calculator priority (low to high)
        /// Features from lower priority calculators appear first
        /// </summary>
        public List<string> GetFeatureNames()
        {
            var orderedFeatures = new List<string>();

            // Sort calculators by priority (ascending)
            var sortedCalculators = _calculators
                .Where(c => c.IsEnabled)
                .OrderBy(c => c.Priority);

            foreach (var calculator in sortedCalculators)
            {
            }

            return orderedFeatures;
        }


        /// <summary>
        /// Process incoming tick data through all aggregators
        /// </summary>
        public void ProcessTick(TickData tick)
        {
            foreach (var aggregator in _aggregators.Values)
            {
                aggregator.AddTick(tick);
            }
        }

        /// <summary>
        /// Calculate all features for current state
        /// Returns feature vector optimized for 3-class prediction
        /// </summary>
        public FeatureVector CalculateFeatures(DateTime timestamp)
        {
            var output = new FeatureVector
            {
                Timestamp = timestamp,
                MarketState = DetermineMarketState()
            };

            // Sort calculators by priority
            var sortedCalculators = _calculators
                .Where(c => c.IsEnabled)
                .OrderBy(c => c.Priority)
                .ThenBy(c => c.Name);

            foreach (var calculator in sortedCalculators)
            {
                try
                {
                    var aggregator = GetAggregator(calculator.Timeframe);
                    if (aggregator == null) continue;

                    var bars = aggregator.GetHistoricalBars(500);
                    if (bars.Count < 50) continue;  // Need minimum history

                    // Calculate features
                    calculator.Calculate(output, bars, bars.Count - 1);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error in {calculator.Name}: {ex.Message}");
                }
            }

            // Validate and clean
            ValidateFeatures(output);

            return output;
        }

        /// <summary>
        /// Validate and clean feature values
        /// </summary>
        private void ValidateFeatures(FeatureVector features)
        {
            var keysToCheck = features.Features.Keys.ToList();

            foreach (var key in keysToCheck)
            {
                var value = features.Features[key];

                // Handle NaN and Infinity
                if (double.IsNaN(value) || double.IsInfinity(value))
                {
                    features.Features[key] = 0.0;
                    Console.WriteLine($"Warning: Invalid value in {key}, set to 0");
                }

                // Clip extreme values
                if (Math.Abs(value) > 10)
                {
                    features.Features[key] = Math.Sign(value) * 10;
                }
            }
        }

        /// <summary>
        /// Determine current market state for adaptive processing
        /// </summary>
        private MarketState DetermineMarketState()
        {
            // Get latest bars from M5 timeframe
            var aggregator = GetAggregator(TimeSpan.FromMinutes(5));
            if (aggregator == null) return MarketState.Normal;

            var bars = aggregator.GetHistoricalBars(20);
            if (bars.Count < 20) return MarketState.Normal;

            // Simple state detection based on volatility and volume
            var avgVolume = bars.Take(19).Average(b => b.TickVolume);
            var currentVolume = bars[0].TickVolume;
            var avgRange = bars.Take(19).Average(b => (double)(b.High - b.Low));
            var currentRange = (double)(bars[0].High - bars[0].Low);

            if (currentVolume > avgVolume * 2 || currentRange > avgRange * 2)
                return MarketState.HighActivity;
            if (currentVolume < avgVolume * 0.5 && currentRange < avgRange * 0.5)
                return MarketState.LowActivity;

            return MarketState.Normal;
        }

        public IBarAggregator? GetAggregator(TimeSpan timeframe)
        {
            return _aggregators.TryGetValue(timeframe, out var aggregator) ? aggregator : null;
        }
    }
}