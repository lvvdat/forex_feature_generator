namespace ForexFeatureGenerator.Pipeline
{
    public class FeatureConfiguration
    {
        public Dictionary<string, bool> EnabledFeatures { get; set; } = new();

        public void EnableFeature(string calculatorName) => EnabledFeatures[calculatorName] = true;
        public void DisableFeature(string calculatorName) => EnabledFeatures[calculatorName] = false;
        public bool IsEnabled(string calculatorName) => EnabledFeatures.GetValueOrDefault(calculatorName, true);

        public static FeatureConfiguration CreateDefault()
        {
            return new FeatureConfiguration
            {
                EnabledFeatures = new Dictionary<string, bool>
                {
                    { "M1_Microstructure", true },
                    { "M1_Momentum", true },
                    { "M1_Volatility", true },
                    { "M5_Trend", true },
                    { "M5_Oscillators", true }
                }
            };
        }

        public void SaveToJson(string path)
        {
            var json = System.Text.Json.JsonSerializer.Serialize(this, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true
            });
            File.WriteAllText(path, json);
        }

        public static FeatureConfiguration LoadFromJson(string path)
        {
            var json = File.ReadAllText(path);
            return System.Text.Json.JsonSerializer.Deserialize<FeatureConfiguration>(json)
                ?? CreateDefault();
        }
    }
}
