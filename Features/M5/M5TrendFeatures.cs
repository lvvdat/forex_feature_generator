using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.M5
{
    public class M5TrendFeatures : BaseFeatureCalculator
    {
        public override string Name => "M5_Trend";
        public override string Category => "Trend";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(5);
        public override int Priority => 20;

        private readonly RollingWindow<double> _slopes = new(50);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 1) return;

            var bar = bars[currentIndex];
            var close = (double)bar.Close;

            // ===== MOVING AVERAGES =====
            if (currentIndex >= 9)
            {
                var ema9 = EMA(bars, 9, currentIndex);
                output.AddFeature("fg3_ema_9", ema9);

                // EMA 9 Slope
                if (currentIndex >= 14)
                {
                    var ema9_prev = EMA(bars, 9, currentIndex - 5);
                    var slope9 = SafeDiv(ema9 - ema9_prev, ema9_prev) * 10000;
                    output.AddFeature("fg3_ema9_slope", slope9);
                    _slopes.Add(slope9);
                }
            }

            if (currentIndex >= 20)
            {
                var sma20 = SMA(bars, 20, currentIndex);
                output.AddFeature("fg3_sma_20", sma20);
            }

            if (currentIndex >= 21)
            {
                var ema21 = EMA(bars, 21, currentIndex);
                output.AddFeature("fg3_ema_21", ema21);

                // EMA 21 Slope
                if (currentIndex >= 26)
                {
                    var ema21_prev = EMA(bars, 21, currentIndex - 5);
                    var slope21 = SafeDiv(ema21 - ema21_prev, ema21_prev) * 10000;
                    output.AddFeature("fg3_ema21_slope", slope21);
                }

                // EMA Crossover 9/21
                if (currentIndex >= 21)
                {
                    var ema9 = EMA(bars, 9, currentIndex);
                    var ema9_prev = EMA(bars, 9, currentIndex - 1);
                    var ema21_prev = EMA(bars, 21, currentIndex - 1);

                    var crossover = 0.0;
                    if (ema9 > ema21 && ema9_prev <= ema21_prev) crossover = 1.0;
                    else if (ema9 < ema21 && ema9_prev >= ema21_prev) crossover = -1.0;

                    output.AddFeature("fg3_ema_cross_9_21", crossover);
                }
            }

            if (currentIndex >= 50)
            {
                var ema50 = EMA(bars, 50, currentIndex);
                var sma50 = SMA(bars, 50, currentIndex);

                output.AddFeature("fg3_ema_50", ema50);
                output.AddFeature("fg3_sma_50", sma50);
                output.AddFeature("fg3_price_above_ema50", close > ema50 ? 1.0 : 0.0);

                // EMA Crossover 21/50
                if (currentIndex >= 50)
                {
                    var ema21 = EMA(bars, 21, currentIndex);
                    var ema21_prev = EMA(bars, 21, currentIndex - 1);
                    var ema50_prev = EMA(bars, 50, currentIndex - 1);

                    var crossover = 0.0;
                    if (ema21 > ema50 && ema21_prev <= ema50_prev) crossover = 1.0;
                    else if (ema21 < ema50 && ema21_prev >= ema50_prev) crossover = -1.0;

                    output.AddFeature("fg3_ema_cross_21_50", crossover);
                }
            }

            // ===== TREND STRENGTH =====
            if (currentIndex >= 50)
            {
                var ema9 = EMA(bars, 9, currentIndex);
                var ema21 = EMA(bars, 21, currentIndex);
                var ema50 = EMA(bars, 50, currentIndex);

                // EMA Alignment (perfect trend alignment)
                var alignment = 0.0;
                if (ema9 > ema21 && ema21 > ema50) alignment = 1.0;
                else if (ema9 < ema21 && ema21 < ema50) alignment = -1.0;

                output.AddFeature("fg3_ema_alignment", alignment);

                // MA Distance (spread between MAs)
                var maDistance = SafeDiv(Math.Abs(ema9 - ema50), ema50) * 10000;
                output.AddFeature("fg3_ma_distance", maDistance);

                // Trend Strength
                var trendStrength = SafeDiv(Math.Abs(ema21 - ema50), ema50) * 10000;
                output.AddFeature("fg3_trend_strength", trendStrength);

                // Trend Consistency (how many bars in same direction)
                var consistency = 0;
                for (int i = currentIndex - 9; i <= currentIndex; i++)
                {
                    if (bars[i].Close > bars[i].Open) consistency++;
                    else consistency--;
                }
                output.AddFeature("fg3_trend_consistency", consistency / 10.0);
            }

            // ===== PRICE SLOPE =====
            if (currentIndex >= 10)
            {
                var priceSlope = CalculatePriceSlope(bars, currentIndex, 10);
                output.AddFeature("fg3_price_slope", priceSlope * 10000);

                // Slope Acceleration
                if (_slopes.Count >= 2)
                {
                    var acceleration = _slopes[0] - _slopes[1];
                    output.AddFeature("fg3_slope_acceleration", acceleration);
                }
                else
                {
                    output.AddFeature("fg3_slope_acceleration", 0.0);
                }
            }

            // ===== KELTNER CHANNEL =====
            if (currentIndex >= 20)
            {
                var ema20 = EMA(bars, 20, currentIndex);
                var atr = ATR(bars, 10, currentIndex);

                var keltnerUpper = ema20 + 2 * atr;
                var keltnerLower = ema20 - 2 * atr;

                output.AddFeature("fg3_keltner_upper", keltnerUpper);
                output.AddFeature("fg3_keltner_lower", keltnerLower);
                output.AddFeature("fg3_keltner_position", SafeDiv(close - keltnerLower, keltnerUpper - keltnerLower));
            }

            // ===== DONCHIAN CHANNEL =====
            if (currentIndex >= 20)
            {
                double highest = double.MinValue;
                double lowest = double.MaxValue;

                for (int i = currentIndex - 19; i <= currentIndex; i++)
                {
                    highest = Math.Max(highest, (double)bars[i].High);
                    lowest = Math.Min(lowest, (double)bars[i].Low);
                }

                output.AddFeature("fg3_donchian_upper", highest);
                output.AddFeature("fg3_donchian_lower", lowest);
                output.AddFeature("fg3_donchian_width", SafeDiv(highest - lowest, close) * 10000);
            }
        }

        public override void Reset()
        {
            _slopes.Clear();
        }
    }
}
