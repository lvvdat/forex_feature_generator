using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ForexFeatureGenerator.Core.Models
{
    /// <summary>
    /// Market state enumeration
    /// </summary>
    public enum MarketState
    {
        LowActivity = -1,
        Normal = 0,
        HighActivity = 1
    }

    public class FeatureVector
    {
        public DateTime Timestamp { get; set; }
        public Dictionary<string, double> Features { get; set; } = new();

        public MarketState MarketState { get; set; }
        public double PredictionConfidence { get; set; }
        public Dictionary<string, double> CategoryScores { get; set; } = new();

        public double GetFeature(string name)
        {
            return Features.TryGetValue(name, out var value) ? value : 0.0;
        }
        public bool TryGetFeature(string name, out double value)
        {
            return Features.TryGetValue(name, out value);
        }

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
