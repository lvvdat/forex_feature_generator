namespace ForexFeatureGenerator.Core.Models
{
    public class FeatureVector
    {
        public DateTime Timestamp { get; set; }
        public Dictionary<string, double> Features { get; set; } = new();

        public void AddFeature(string name, double value)
        {
            // Handle NaN and Infinity
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                Features[name] = 0.0;
                return;
            }
            Features[name] = value;
        }
    }
}
