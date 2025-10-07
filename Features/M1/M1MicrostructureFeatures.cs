using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.M1
{
    public class M1MicrostructureFeatures : BaseFeatureCalculator
    {
        public override string Name => "M1_Microstructure";
        public override string Category => "Microstructure";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(1);
        public override int Priority => 1; // Calculate early

        private readonly RollingWindow<double> _spreadHistory = new(60);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 0 || bars.Count == 0) return;

            var bar = bars[currentIndex];

            // Candle characteristics
            var range = (double)(bar.High - bar.Low);
            output.AddFeature("fg1_normalized_range", SafeDiv(range, (double)bar.Close));

            var bodySize = Math.Abs((double)(bar.Close - bar.Open));
            output.AddFeature("fg1_body_size", SafeDiv(bodySize, (double)bar.Close));

            var upperWick = (double)(bar.High - Math.Max(bar.Open, bar.Close));
            var lowerWick = (double)(Math.Min(bar.Open, bar.Close) - bar.Low);

            output.AddFeature("fg1_upper_wick", SafeDiv(upperWick, range));
            output.AddFeature("fg1_lower_wick", SafeDiv(lowerWick, range));
        }

        public override void Reset()
        {
            _spreadHistory.Clear();
        }
    }

}
