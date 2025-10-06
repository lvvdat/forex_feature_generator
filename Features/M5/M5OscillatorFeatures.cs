using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;

namespace ForexFeatureGenerator.Features.M5
{
    public class M5OscillatorFeatures : BaseFeatureCalculator
    {
        public override string Name => "M5_Oscillators";
        public override string Category => "Momentum";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(5);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return;

            // RSI(9)
            var rsi9 = CalculateRSI(bars, 9, currentIndex);
            output.AddFeature("m5_rsi_9", rsi9);

            // RSI(14)
            if (currentIndex >= 20)
            {
                var rsi14 = CalculateRSI(bars, 14, currentIndex);
                output.AddFeature("m5_rsi_14", rsi14);
            }

            // Bollinger Bands
            var sma20 = SMA(bars, 20, currentIndex);
            var std20 = StdDev(bars, 20, currentIndex);
            var bbUpper = sma20 + 2 * std20;
            var bbLower = sma20 - 2 * std20;

            var bbWidth = SafeDiv(bbUpper - bbLower, sma20);
            output.AddFeature("m5_bb_width", bbWidth);

            var close = (double)bars[currentIndex].Close;
            var bbPosition = SafeDiv(close - bbLower, bbUpper - bbLower);
            output.AddFeature("m5_bb_position", bbPosition);
        }

        private double CalculateRSI(IReadOnlyList<OhlcBar> bars, int period, int currentIndex)
        {
            if (currentIndex < period) return 50;

            double gains = 0;
            double losses = 0;

            for (int i = currentIndex - period + 1; i <= currentIndex; i++)
            {
                var change = (double)(bars[i].Close - bars[i - 1].Close);
                if (change > 0)
                    gains += change;
                else
                    losses += Math.Abs(change);
            }

            var avgGain = gains / period;
            var avgLoss = losses / period;

            if (avgLoss < 1e-10) return 100;

            var rs = avgGain / avgLoss;
            return 100 - (100 / (1 + rs));
        }
    }

}
