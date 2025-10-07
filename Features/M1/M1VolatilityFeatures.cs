using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.M1
{
    public class M1VolatilityFeatures : BaseFeatureCalculator
    {
        public override string Name => "M1_Volatility";
        public override string Category => "Volatility";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(1);
        public override int Priority => 4; // Calculate early

        private readonly RollingWindow<double> _trueRanges = new(20);
        private readonly RollingWindow<double> _logReturns = new(20);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 1) return;

            var bar = bars[currentIndex];
            var prevBar = bars[currentIndex - 1];

            // True Range
            var tr = Math.Max(
                (double)(bar.High - bar.Low),
                Math.Max(
                    Math.Abs((double)(bar.High - prevBar.Close)),
                    Math.Abs((double)(bar.Low - prevBar.Close))
                )
            );

            _trueRanges.Add(tr);
            output.AddFeature("fg1_tr_current", tr);

            // ATR(10)
            if (_trueRanges.Count >= 10)
            {
                var atr = _trueRanges.GetValues().Take(10).Average();
                output.AddFeature("fg1_atr_10", atr);
            }

            // Log returns for Realized Volatility
            var logReturn = Math.Log((double)bar.Close / (double)prevBar.Close);
            _logReturns.Add(logReturn);

            // Realized Volatility (20-period)
            if (_logReturns.Count >= 20)
            {
                var sumSquares = _logReturns.GetValues().Take(20).Sum(r => r * r);
                var rv = Math.Sqrt(sumSquares);
                output.AddFeature("fg1_rv_20", rv);
            }
            else
            {
                output.AddFeature("fg1_rv_20", 0.0);
            }
        }

        public override void Reset()
        {
            _trueRanges.Clear();
            _logReturns.Clear();
        }
    }
}
