using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;

namespace ForexFeatureGenerator.Features.M5
{
    public class M5OscillatorFeatures : BaseFeatureCalculator
    {
        public override string Name => "M5_Oscillators";
        public override string Category => "Momentum";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(5);
        public override int Priority => 21;

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return;

            // RSI(9)
            var rsi9 = CalculateRSI(bars, 9, currentIndex);
            output.AddFeature("fg3_rsi_9", rsi9);

            // RSI(14)
            if (currentIndex >= 20)
            {
                var rsi14 = CalculateRSI(bars, 14, currentIndex);
                output.AddFeature("fg3_rsi_14", rsi14);
            }

            // Bollinger Bands
            var sma20 = SMA(bars, 20, currentIndex);
            var std20 = StdDev(bars, 20, currentIndex);
            var bbUpper = sma20 + 2 * std20;
            var bbLower = sma20 - 2 * std20;

            var bbWidth = SafeDiv(bbUpper - bbLower, sma20);
            output.AddFeature("fg3_bb_width", bbWidth);

            var close = (double)bars[currentIndex].Close;
            var bbPosition = SafeDiv(close - bbLower, bbUpper - bbLower);
            output.AddFeature("fg3_bb_position", bbPosition);
        }
    }
}
