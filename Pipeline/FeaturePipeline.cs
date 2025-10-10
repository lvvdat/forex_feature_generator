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

            // Apply feature post-processing
            PostProcessFeatures(output);

            // Calculate meta-features
            AddMetaFeatures(output);

            // Validate and clean
            ValidateFeatures(output);

            return output;
        }

        /// <summary>
        /// Post-process features for optimal prediction
        /// </summary>
        private void PostProcessFeatures(FeatureVector features)
        {
            // 1. Create interaction features
            CreateInteractionFeatures(features);

            // 2. Apply non-linear transformations
            ApplyNonLinearTransformations(features);
        }

        /// <summary>
        /// Create interaction features between important indicators
        /// </summary>
        private void CreateInteractionFeatures(FeatureVector features)
        {
            // Momentum × Volume interaction
            if (features.TryGetFeature("dir_momentum_z5", out var momentum) &&
                features.TryGetFeature("micro_flow_imbalance", out var flowImbalance))
            {
                features.AddFeature("08_interaction_momentum_flow", momentum * flowImbalance);
            }
            else
            {
                features.AddFeature("08_interaction_momentum_flow", 0.0);
            }

            // Trend × Volatility interaction
            if (features.TryGetFeature("trend_mtf_strength", out var trendStrength) &&
                features.TryGetFeature("vol_regime_type", out var volRegime))
            {
                features.AddFeature("08_interaction_trend_volatility", trendStrength * (1 - Math.Abs(volRegime)));
            }
            else
            {
                features.AddFeature("08_interaction_trend_volatility", 0.0);
            }

            // Technical × Microstructure interaction
            if (features.TryGetFeature("tech_master_signal", out var techSignal) &&
                features.TryGetFeature("micro_master_signal", out var microSignal))
            {
                features.AddFeature("08_interaction_tech_micro", (techSignal + microSignal) / 2);
            }
            else
            {
                features.AddFeature("08_interaction_tech_micro", 0.0);
            }

            // Regime × Direction interaction
            if (features.TryGetFeature("regime_type", out var regime) &&
                features.TryGetFeature("dir_composite_primary", out var direction))
            {
                // Adjust direction based on regime
                var regimeAdjusted = regime == 1 ? direction * 1.2 :  // Trending: amplify
                                    regime == 0 ? direction * -0.5 :  // Range: fade
                                    direction * 0.8;                  // Volatile: reduce
                features.AddFeature("08_interaction_regime_direction", Math.Max(-1, Math.Min(1, regimeAdjusted)));
            }
            else
            {
                features.AddFeature("08_interaction_regime_direction", 0.0);
            }
        }

        /// <summary>
        /// Apply non-linear transformations for better separability
        /// </summary>
        private void ApplyNonLinearTransformations(FeatureVector features)
        {
            // Polynomial features for key indicators
            var keyFeatures = new[] { "dir_momentum_z5", "micro_flow_imbalance", "tech_oscillator_composite" };

            foreach (var key in keyFeatures)
            {
                if (features.TryGetFeature(key, out var value))
                {
                    features.AddFeature($"08_{key}_squared", value * value);
                    features.AddFeature($"08_{key}_cubed", value * value * value);
                }
                else
                {
                    features.AddFeature($"08_{key}_squared", 0.0);
                    features.AddFeature($"08_{key}_cubed", 0.0);
                }

            }

            // Log transformations for volume features
            var volumeFeatures = features.Features
                .Where(f => f.Key.Contains("volume") || f.Key.Contains("tick"))
                .ToList();

            foreach (var feature in volumeFeatures)
            {
                if (feature.Value > 0)
                {
                    features.AddFeature($"08_{feature.Key}_log", Math.Log(1 + feature.Value));
                }
                else
                {
                    features.AddFeature($"08_{feature.Key}_log", 0.0);
                }
            }
        }

        /// <summary>
        /// Add meta-features that capture overall market state
        /// </summary>
        private void AddMetaFeatures(FeatureVector features)
        {
            // Feature agreement score
            var signals = features.Features
                .Where(f => f.Key.Contains("_signal"))
                .Select(f => f.Value)
                .ToList();

            if (signals.Any())
            {
                var agreement = CalculateAgreement(signals);
                features.AddFeature("08_meta_signal_agreement", agreement);

                // Signal strength (average absolute value)
                var strength = signals.Select(Math.Abs).Average();
                features.AddFeature("08_meta_signal_strength", strength);
            }
            else
            {
                features.AddFeature("08_meta_signal_agreement", 0.0);
                features.AddFeature("08_meta_signal_strength", 0.0);
            }

            // Feature quality score
            var qualityScore = CalculateFeatureQuality(features);
            features.AddFeature("08_meta_feature_quality", qualityScore);

            // Prediction confidence based on feature completeness
            var confidence = CalculatePredictionConfidence(features);
            features.AddFeature("08_meta_prediction_confidence", confidence);
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

        private double CalculateAgreement(List<double> signals)
        {
            if (!signals.Any()) return 0;

            var positive = signals.Count(s => s > 0.1);
            var negative = signals.Count(s => s < -0.1);
            var total = signals.Count;

            var agreement = Math.Max(positive, negative) / (double)total;
            return positive > negative ? agreement : -agreement;
        }

        private double CalculateFeatureQuality(FeatureVector features)
        {
            // Quality based on feature completeness and validity
            var totalPossible = _calculators.Count * 20;  // Approximate features per calculator
            var actualFeatures = features.Features.Count;
            var completeness = Math.Min(1.0, actualFeatures / (double)totalPossible);

            var validRatio = features.Features.Values.Count(v => Math.Abs(v) > 0.01) / (double)actualFeatures;

            return (completeness + validRatio) / 2;
        }

        private double CalculatePredictionConfidence(FeatureVector features)
        {
            // Confidence based on signal strength and agreement
            var primarySignals = new[]
            {
                "dir_composite_primary",
                "tech_master_signal",
                "micro_master_signal",
                "regime_master_signal"
            };

            var signals = primarySignals
                .Where(s => features.TryGetFeature(s, out _))
                .Select(s => features.GetFeature(s))
                .ToList();

            if (!signals.Any()) return 0;

            var avgStrength = signals.Select(Math.Abs).Average();
            var agreement = Math.Abs(CalculateAgreement(signals));

            return (avgStrength + agreement) / 2;
        }
    }
}