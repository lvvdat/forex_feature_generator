using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;

namespace ForexFeatureGenerator.Features.M1
{
    public class M1MomentumFeatures : BaseFeatureCalculator
    {
        public override string Name => "M1_Momentum";
        public override string Category => "Momentum";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(1);
        public override int Priority => 2; // Calculate early

        private double _prevRoc5 = 0;

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 10) return;

            // EMAs
            var ema5 = EMA(bars, 5, currentIndex);
            var ema8 = EMA(bars, 8, currentIndex);

            output.AddFeature("fg1_ema_5", ema5);
            output.AddFeature("fg1_ema_8", ema8);
            output.AddFeature("fg1_ema_ratio", SafeDiv(ema5, ema8, 1.0));

            // Rate of Change
            if (currentIndex >= 5)
            {
                var close = (double)bars[currentIndex].Close;
                var close5 = (double)bars[currentIndex - 5].Close;
                var roc5 = SafeDiv(close - close5, close5) * 10000;
                output.AddFeature("fg1_roc_5", roc5);

                // Acceleration (change in ROC)
                var acceleration = roc5 - _prevRoc5;
                output.AddFeature("fg1_price_acceleration", acceleration);
                _prevRoc5 = roc5;
            }

            if (currentIndex >= 10)
            {
                var close = (double)bars[currentIndex].Close;
                var close10 = (double)bars[currentIndex - 10].Close;
                var roc10 = SafeDiv(close - close10, close10) * 10000;
                output.AddFeature("fg1_roc_10", roc10);
            }
        }

        public override void Reset()
        {
            _prevRoc5 = 0;
        }
    }

}
