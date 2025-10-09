using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Core;
using ForexFeatureGenerator.Core.Infrastructure;
using ForexFeatureGenerator.Pipeline;

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
        private readonly FeatureStatistics _statistics = new();

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

            // Additional enhanced calculators can be added here
            // RegisterCalculator(new PatternRecognitionFeatures());
            // RegisterCalculator(new MachineLearningFeatures());
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

                    // Track feature statistics
                    _statistics.UpdateStatistics(calculator.Name, output);
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
            // 1. Normalize features to consistent range
            NormalizeFeatures(features);

            // 2. Create interaction features
            CreateInteractionFeatures(features);

            // 3. Apply non-linear transformations
            ApplyNonLinearTransformations(features);

            // 4. Feature selection based on importance
            //SelectImportantFeatures(features);
        }

        /// <summary>
        /// Normalize features to [-1, 1] range
        /// </summary>
        private void NormalizeFeatures(FeatureVector features)
        {
            var featuresToNormalize = features.Features
                .Where(f => !f.Key.Contains("_signal") && !f.Key.Contains("_normalized"))
                .ToList();

            foreach (var feature in featuresToNormalize)
            {
                var stats = _statistics.GetFeatureStats(feature.Key);
                if (stats != null && stats.StdDev > 0)
                {
                    // Z-score normalization
                    var normalized = (feature.Value - stats.Mean) / stats.StdDev;
                    // Clip to [-3, 3] and scale to [-1, 1]
                    normalized = Math.Max(-3, Math.Min(3, normalized)) / 3;
                    features.Features[feature.Key] = normalized;
                }
            }
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
                features.AddFeature("interaction_momentum_flow", momentum * flowImbalance);
            }
            else
            {
                features.AddFeature("interaction_momentum_flow", 0.0);
            }

            // Trend × Volatility interaction
            if (features.TryGetFeature("trend_mtf_strength", out var trendStrength) &&
                features.TryGetFeature("vol_regime_type", out var volRegime))
            {
                features.AddFeature("interaction_trend_volatility", trendStrength * (1 - Math.Abs(volRegime)));
            }
            else
            {
                features.AddFeature("interaction_trend_volatility", 0.0);
            }

            // Technical × Microstructure interaction
            if (features.TryGetFeature("tech_master_signal", out var techSignal) &&
                features.TryGetFeature("micro_master_signal", out var microSignal))
            {
                features.AddFeature("interaction_tech_micro", (techSignal + microSignal) / 2);
            }
            else
            {
                features.AddFeature("interaction_tech_micro", 0.0);
            }

            // Regime × Direction interaction
            if (features.TryGetFeature("regime_type", out var regime) &&
                features.TryGetFeature("dir_composite_primary", out var direction))
            {
                // Adjust direction based on regime
                var regimeAdjusted = regime == 1 ? direction * 1.2 :  // Trending: amplify
                                    regime == 0 ? direction * -0.5 :  // Range: fade
                                    direction * 0.8;                   // Volatile: reduce
                features.AddFeature("interaction_regime_direction", Math.Max(-1, Math.Min(1, regimeAdjusted)));
            }
            else
            {
                features.AddFeature("interaction_regime_direction", 0.0);
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
                    features.AddFeature($"{key}_squared", value * value);
                    features.AddFeature($"{key}_cubed", value * value * value);
                }
                else
                {
                    features.AddFeature($"{key}_squared", 0.0);
                    features.AddFeature($"{key}_cubed", 0.0);
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
                    features.AddFeature($"{feature.Key}_log", Math.Log(1 + feature.Value));
                }
                else
                {
                    features.AddFeature($"{feature.Key}_log", 0.0);
                }
            }
        }

        /// <summary>
        /// Select most important features based on statistics
        /// </summary>
        private void SelectImportantFeatures(FeatureVector features)
        {
            // Remove features with low variance (uninformative)
            var lowVarianceFeatures = _statistics.GetLowVarianceFeatures(threshold: 0.01);
            foreach (var feature in lowVarianceFeatures)
            {
                features.RemoveFeature(feature);
            }

            // Remove highly correlated features (redundant)
            var correlatedPairs = _statistics.GetHighlyCorrelatedFeatures(threshold: 0.95);
            foreach (var pair in correlatedPairs)
            {
                // Keep the one with higher importance score
                var importance1 = _statistics.GetFeatureImportance(pair.Item1);
                var importance2 = _statistics.GetFeatureImportance(pair.Item2);

                features.RemoveFeature(importance1 > importance2 ? pair.Item2 : pair.Item1);
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
                features.AddFeature("meta_signal_agreement", agreement);

                // Signal strength (average absolute value)
                var strength = signals.Select(Math.Abs).Average();
                features.AddFeature("meta_signal_strength", strength);
            }
            else
            {
                features.AddFeature("meta_signal_agreement", 0.0);
                features.AddFeature("meta_signal_strength", 0.0);
            }

            // Feature quality score
            var qualityScore = CalculateFeatureQuality(features);
            features.AddFeature("meta_feature_quality", qualityScore);

            // Prediction confidence based on feature completeness
            var confidence = CalculatePredictionConfidence(features);
            features.AddFeature("meta_prediction_confidence", confidence);

            // Market complexity indicator
            var complexity = CalculateMarketComplexity(features);
            features.AddFeature("meta_market_complexity", complexity);
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

        private double CalculateMarketComplexity(FeatureVector features)
        {
            var complexity = 0.0;
            var count = 0;

            // High complexity if: high volatility, low efficiency, regime transitions
            if (features.TryGetFeature("vol_regime_type", out var volRegime))
            {
                complexity += Math.Abs(volRegime);
                count++;
            }

            if (features.TryGetFeature("trend_efficiency", out var efficiency))
            {
                complexity += 1 - efficiency;
                count++;
            }

            if (features.TryGetFeature("regime_transition_prob", out var transition))
            {
                complexity += transition;
                count++;
            }

            return count > 0 ? complexity / count : 0.5;
        }

        /// <summary>
        /// Get feature importance ranking
        /// </summary>
        public Dictionary<string, double> GetFeatureImportance()
        {
            return _statistics.GetAllFeatureImportance()
                .OrderByDescending(f => f.Value)
                .ToDictionary(f => f.Key, f => f.Value);
        }

        /// <summary>
        /// Get feature statistics for analysis
        /// </summary>
        public FeatureStatistics GetStatistics()
        {
            return _statistics;
        }

        /// <summary>
        /// Reset all calculators and statistics
        /// </summary>
        public void Reset()
        {
            foreach (var calculator in _calculators)
            {
                calculator.Reset();
            }
            _statistics.Reset();
        }
    }

    /// <summary>
    /// Feature statistics tracking
    /// </summary>
    public class FeatureStatistics
    {
        private readonly Dictionary<string, RunningStats> _stats = new();
        private readonly Dictionary<string, double> _importance = new();
        private readonly Dictionary<(string, string), double> _correlations = new();

        public void UpdateStatistics(string calculatorName, FeatureVector features)
        {
            foreach (var feature in features.Features)
            {
                var key = feature.Key;
                if (!_stats.ContainsKey(key))
                {
                    _stats[key] = new RunningStats();
                }

                _stats[key].Update(feature.Value);

                // Update importance based on signal strength
                if (key.Contains("signal") || key.Contains("composite"))
                {
                    _importance[key] = (_importance.GetValueOrDefault(key, 0) * 0.99) +
                                      Math.Abs(feature.Value) * 0.01;
                }
            }
        }

        public RunningStats? GetFeatureStats(string feature)
        {
            return _stats.GetValueOrDefault(feature);
        }

        public double GetFeatureImportance(string feature)
        {
            return _importance.GetValueOrDefault(feature, 0.5);
        }

        public List<string> GetLowVarianceFeatures(double threshold)
        {
            return _stats
                .Where(s => s.Value.Variance < threshold)
                .Select(s => s.Key)
                .ToList();
        }

        public List<(string, string)> GetHighlyCorrelatedFeatures(double threshold)
        {
            return _correlations
                .Where(c => Math.Abs(c.Value) > threshold)
                .Select(c => c.Key)
                .ToList();
        }

        public Dictionary<string, double> GetAllFeatureImportance()
        {
            return new Dictionary<string, double>(_importance);
        }

        public void Reset()
        {
            _stats.Clear();
            _importance.Clear();
            _correlations.Clear();
        }

        public class RunningStats
        {
            private double _m = 0;
            private double _s = 0;
            private int _n = 0;

            public double Mean => _n > 0 ? _m : 0;
            public double Variance => _n > 1 ? _s / (_n - 1) : 0;
            public double StdDev => Math.Sqrt(Variance);
            public int Count => _n;

            public void Update(double value)
            {
                _n++;
                if (_n == 1)
                {
                    _m = value;
                    _s = 0;
                }
                else
                {
                    var oldM = _m;
                    _m = oldM + (value - oldM) / _n;
                    _s = _s + (value - oldM) * (value - _m);
                }
            }
        }
    }
}