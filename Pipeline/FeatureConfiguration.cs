namespace ForexFeatureGenerator.Pipeline
{
    public enum FeatureSelectionMode
    {
        All,        // Use all features
        Manual,     // Use manually selected features
        Adaptive,   // Dynamically select based on importance
        Optimized   // Use pre-optimized feature set
    }

    /// <summary>
    /// Feature configuration optimized for 3-class prediction
    /// </summary>
    public class FeatureConfiguration
    {
        public Dictionary<string, bool> EnabledFeatures { get; set; } = new();
        public Dictionary<string, double> FeatureWeights { get; set; } = new();
        public FeatureSelectionMode SelectionMode { get; set; } = FeatureSelectionMode.Adaptive;

        public static FeatureConfiguration CreateOptimized3Class()
        {
            return new FeatureConfiguration
            {
                EnabledFeatures = new Dictionary<string, bool>
                {
                    { "Directional", true },
                    { "MarketRegimeContext", true },
                    { "MicrostructureOrderFlow", true },
                    { "TechnicalIndicators", true }
                },
                FeatureWeights = new Dictionary<string, double>
                {
                    { "dir_composite_primary", 1.5 },
                    { "micro_flow_composite", 1.2 },
                },
                SelectionMode = FeatureSelectionMode.Adaptive
            };
        }

        public bool IsFeatureEnabled(string name)
        {
            return EnabledFeatures.GetValueOrDefault(name, true);
        }

        public double GetFeatureWeight(string name)
        {
            return FeatureWeights.GetValueOrDefault(name, 1.0);
        }
    }
}
