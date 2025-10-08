using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Pipeline
{
    public class FeaturePipeline
    {
        private readonly Dictionary<TimeSpan, IBarAggregator> _aggregators = new();
        private readonly List<IFeatureCalculator> _calculators = new();
        private readonly FeatureConfiguration _config;

        public FeaturePipeline(FeatureConfiguration config)
        {
            _config = config;
        }

        public void RegisterAggregator(TimeSpan timeframe, int historySize = 500)
        {
            _aggregators[timeframe] = new BarAggregator(timeframe, historySize);
        }

        public IBarAggregator GetAggregator(TimeSpan timeframe)
        {
            return _aggregators.TryGetValue(timeframe, out var agg) ? agg : null;
        }

        public void RegisterCalculator(IFeatureCalculator calculator)
        {
            calculator.IsEnabled = _config.IsEnabled(calculator.Name);
            _calculators.Add(calculator);
        }

        public void ProcessTick(TickData tick)
        {
            foreach (var aggregator in _aggregators.Values)
            {
                aggregator.AddTick(tick);
            }
        }

        public FeatureVector CalculateFeatures(DateTime timestamp)
        {
            var output = new FeatureVector { Timestamp = timestamp };

            var sortedCalculators = _calculators
                .Where(c => c.IsEnabled)
                .OrderBy(c => c.Priority);

            foreach (var calculator in sortedCalculators)
            {
                if (!_aggregators.TryGetValue(calculator.Timeframe, out var aggregator))
                    continue;

                var bars = aggregator.GetHistoricalBars(500);
                if (bars.Count == 0)
                    continue;

                // Use index based on bars list (0 = most recent)
                calculator.Calculate(output, bars, bars.Count - 1);
            }

            return output;
        }

        public void Reset()
        {
            foreach (var calculator in _calculators)
            {
                calculator.Reset();
            }
        }
    }
}
