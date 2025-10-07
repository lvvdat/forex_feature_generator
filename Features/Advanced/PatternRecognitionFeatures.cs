using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.Advanced
{
    public class PatternRecognitionFeatures : BaseFeatureCalculator
    {
        public override string Name => "PatternRecognition";
        public override string Category => "Patterns";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(1);
        public override int Priority => 12;

        private readonly RollingWindow<int> _patternHistory = new(50);
        private readonly RollingWindow<double> _swingHighs = new(20);
        private readonly RollingWindow<double> _swingLows = new(20);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 3) return;

            var curr = bars[currentIndex];
            var prev1 = bars[currentIndex - 1];
            var prev2 = currentIndex >= 2 ? bars[currentIndex - 2] : null;
            var prev3 = currentIndex >= 3 ? bars[currentIndex - 3] : null;

            // ===== CANDLESTICK PATTERNS =====

            // Bullish Engulfing
            var bullishEngulfing = prev1.Close < prev1.Open && curr.Close > curr.Open &&
                                  curr.Open <= prev1.Close && curr.Close >= prev1.Open ? 1.0 : 0.0;
            output.AddFeature("fg2_pattern_bullish_engulfing", bullishEngulfing);

            // Bearish Engulfing  
            var bearishEngulfing = prev1.Close > prev1.Open && curr.Close < curr.Open &&
                                  curr.Open >= prev1.Close && curr.Close <= prev1.Open ? 1.0 : 0.0;
            output.AddFeature("fg2_pattern_bearish_engulfing", bearishEngulfing);

            // Calculate body and shadow ratios
            var range = curr.High - curr.Low;
            var body = Math.Abs(curr.Close - curr.Open);
            var upperShadow = curr.High - Math.Max(curr.Open, curr.Close);
            var lowerShadow = Math.Min(curr.Open, curr.Close) - curr.Low;

            // Hammer
            var hammer = range > 0 && body / range < 0.3m && lowerShadow / range > 0.6m &&
                        curr.Close > prev1.Close ? 1.0 : 0.0;
            output.AddFeature("fg2_pattern_hammer", hammer);

            // Shooting Star
            var shootingStar = range > 0 && body / range < 0.3m && upperShadow / range > 0.6m &&
                              curr.Close < prev1.Close ? 1.0 : 0.0;
            output.AddFeature("fg2_pattern_shooting_star", shootingStar);

            // Doji (small body)
            var doji = range > 0 && body / range < 0.1m ? 1.0 : 0.0;
            output.AddFeature("fg2_pattern_doji", doji);

            // Three White Soldiers
            var threeWhiteSoldiers = 0.0;
            if (currentIndex >= 2 && prev2 != null)
            {
                threeWhiteSoldiers = curr.Close > curr.Open &&
                                    prev1.Close > prev1.Open &&
                                    prev2.Close > prev2.Open &&
                                    curr.Close > prev1.Close &&
                                    prev1.Close > prev2.Close ? 1.0 : 0.0;
            }
            output.AddFeature("fg2_pattern_three_white_soldiers", threeWhiteSoldiers);

            // Three Black Crows
            var threeBlackCrows = 0.0;
            if (currentIndex >= 2 && prev2 != null)
            {
                threeBlackCrows = curr.Close < curr.Open &&
                                 prev1.Close < prev1.Open &&
                                 prev2.Close < prev2.Open &&
                                 curr.Close < prev1.Close &&
                                 prev1.Close < prev2.Close ? 1.0 : 0.0;
            }
            output.AddFeature("fg2_pattern_three_black_crows", threeBlackCrows);

            // Morning Star
            var morningStar = 0.0;
            if (currentIndex >= 2 && prev2 != null)
            {
                var firstCandleBearish = prev2.Close < prev2.Open;
                var middleCandleSmall = Math.Abs(prev1.Close - prev1.Open) < (prev2.High - prev2.Low) * 0.3m;
                var thirdCandleBullish = curr.Close > curr.Open && curr.Close > (prev2.Open + prev2.Close) / 2;
                morningStar = firstCandleBearish && middleCandleSmall && thirdCandleBullish ? 1.0 : 0.0;
            }
            output.AddFeature("fg2_pattern_morning_star", morningStar);

            // Evening Star
            var eveningStar = 0.0;
            if (currentIndex >= 2 && prev2 != null)
            {
                var firstCandleBullish = prev2.Close > prev2.Open;
                var middleCandleSmall = Math.Abs(prev1.Close - prev1.Open) < (prev2.High - prev2.Low) * 0.3m;
                var thirdCandleBearish = curr.Close < curr.Open && curr.Close < (prev2.Open + prev2.Close) / 2;
                eveningStar = firstCandleBullish && middleCandleSmall && thirdCandleBearish ? 1.0 : 0.0;
            }
            output.AddFeature("fg2_pattern_evening_star", eveningStar);

            // Bullish Harami
            var bullishHarami = prev1.Close < prev1.Open && curr.Close > curr.Open &&
                               curr.Open > prev1.Close && curr.Close < prev1.Open ? 1.0 : 0.0;
            output.AddFeature("fg2_pattern_bullish_harami", bullishHarami);

            // Bearish Harami
            var bearishHarami = prev1.Close > prev1.Open && curr.Close < curr.Open &&
                               curr.Open < prev1.Close && curr.Close > prev1.Open ? 1.0 : 0.0;
            output.AddFeature("fg2_pattern_bearish_harami", bearishHarami);

            // Tweezer Top
            var tweezerTop = Math.Abs(curr.High - prev1.High) < range * 0.1m &&
                            curr.Close < curr.Open && prev1.Close > prev1.Open ? 1.0 : 0.0;
            output.AddFeature("fg2_pattern_tweezer_top", tweezerTop);

            // Tweezer Bottom
            var tweezerBottom = Math.Abs(curr.Low - prev1.Low) < range * 0.1m &&
                               curr.Close > curr.Open && prev1.Close < prev1.Open ? 1.0 : 0.0;
            output.AddFeature("fg2_pattern_tweezer_bottom", tweezerBottom);

            // Spinning Top
            var spinningTop = range > 0 && body / range < 0.3m &&
                             upperShadow / range > 0.2m && lowerShadow / range > 0.2m ? 1.0 : 0.0;
            output.AddFeature("fg2_pattern_spinning_top", spinningTop);

            // Marubozu (no shadows)
            var marubozu = range > 0 && body / range > 0.95m ? 1.0 : 0.0;
            output.AddFeature("fg2_pattern_marubozu", marubozu);

            // ===== PRICE ACTION PATTERNS =====

            // Track swing highs and lows
            UpdateSwingPoints(bars, currentIndex);

            // Higher High / Lower Low
            var higherHigh = 0.0;
            var lowerLow = 0.0;
            if (_swingHighs.Count >= 2 && _swingLows.Count >= 2)
            {
                higherHigh = _swingHighs[0] > _swingHighs[1] ? 1.0 : 0.0;
                lowerLow = _swingLows[0] < _swingLows[1] ? 1.0 : 0.0;
            }
            output.AddFeature("fg2_pattern_higher_high", higherHigh);
            output.AddFeature("fg2_pattern_lower_low", lowerLow);

            // Double Top/Bottom
            var doubleTop = DetectDoubleTop(bars, currentIndex);
            var doubleBottom = DetectDoubleBottom(bars, currentIndex);
            output.AddFeature("fg2_pattern_double_top", doubleTop);
            output.AddFeature("fg2_pattern_double_bottom", doubleBottom);

            // Head and Shoulders
            var headShoulders = DetectHeadShoulders(bars, currentIndex);
            var inverseHeadShoulders = DetectInverseHeadShoulders(bars, currentIndex);
            output.AddFeature("fg2_pattern_head_shoulders", headShoulders);
            output.AddFeature("fg2_pattern_inverse_head_shoulders", inverseHeadShoulders);

            // Triangle Pattern
            var triangle = DetectTriangle(bars, currentIndex);
            output.AddFeature("fg2_pattern_triangle", triangle);

            // Flag Pattern
            var flag = DetectFlag(bars, currentIndex);
            output.AddFeature("fg2_pattern_flag", flag);

            // Wedge Pattern
            var wedge = DetectWedge(bars, currentIndex);
            output.AddFeature("fg2_pattern_wedge", wedge);

            // ===== PATTERN METRICS =====

            // Pattern Strength (how many patterns detected)
            var patternCount = bullishEngulfing + bearishEngulfing + hammer + shootingStar +
                              doji + morningStar + eveningStar + threeWhiteSoldiers + threeBlackCrows;
            output.AddFeature("fg2_pattern_strength", patternCount);

            // Pattern Confirmation (volume confirmation)
            var volumeConfirmation = 0.0;
            if (patternCount > 0 && curr.TickVolume > prev1.TickVolume * 1.2)
            {
                volumeConfirmation = 1.0;
            }
            output.AddFeature("fg2_pattern_confirmation", volumeConfirmation);

            // Pattern Frequency (recent pattern activity)
            _patternHistory.Add((int)patternCount);
            var patternFrequency = _patternHistory.Count > 0 ?
                _patternHistory.GetValues().Average() : 0.0;
            output.AddFeature("fg2_pattern_frequency", patternFrequency);

            // Pattern Success Rate (simplified - based on immediate price movement)
            var successRate = CalculatePatternSuccessRate(bars, currentIndex, (int)patternCount);
            output.AddFeature("fg2_pattern_success_rate", successRate);
        }

        private void UpdateSwingPoints(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 5) return;

            // Check for swing high (highest point in 5 bars)
            var isSwingHigh = true;
            var isSwingLow = true;
            var midBar = bars[currentIndex - 2];

            for (int i = currentIndex - 4; i <= currentIndex; i++)
            {
                if (i == currentIndex - 2) continue;
                if (bars[i].High >= midBar.High) isSwingHigh = false;
                if (bars[i].Low <= midBar.Low) isSwingLow = false;
            }

            if (isSwingHigh)
                _swingHighs.Add((double)midBar.High);
            if (isSwingLow)
                _swingLows.Add((double)midBar.Low);
        }

        private double DetectDoubleTop(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return 0;

            // Find two peaks with similar heights
            double firstPeak = 0;
            int firstPeakIdx = -1;
            double secondPeak = 0;

            for (int i = currentIndex - 19; i <= currentIndex - 10; i++)
            {
                if ((double)bars[i].High > firstPeak)
                {
                    firstPeak = (double)bars[i].High;
                    firstPeakIdx = i;
                }
            }

            for (int i = currentIndex - 9; i <= currentIndex; i++)
            {
                if ((double)bars[i].High > secondPeak)
                {
                    secondPeak = (double)bars[i].High;
                }
            }

            // Check if peaks are similar (within 0.5%)
            if (firstPeak > 0 && secondPeak > 0)
            {
                var diff = Math.Abs(firstPeak - secondPeak) / firstPeak;
                if (diff < 0.005)
                {
                    // Check for valley between peaks
                    var valley = double.MaxValue;
                    for (int i = firstPeakIdx + 1; i < currentIndex; i++)
                    {
                        valley = Math.Min(valley, (double)bars[i].Low);
                    }

                    if (valley < firstPeak * 0.98) // Valley at least 2% below peaks
                        return 1.0;
                }
            }

            return 0;
        }

        private double DetectDoubleBottom(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return 0;

            // Find two troughs with similar depths
            double firstTrough = double.MaxValue;
            int firstTroughIdx = -1;
            double secondTrough = double.MaxValue;

            for (int i = currentIndex - 19; i <= currentIndex - 10; i++)
            {
                if ((double)bars[i].Low < firstTrough)
                {
                    firstTrough = (double)bars[i].Low;
                    firstTroughIdx = i;
                }
            }

            for (int i = currentIndex - 9; i <= currentIndex; i++)
            {
                if ((double)bars[i].Low < secondTrough)
                {
                    secondTrough = (double)bars[i].Low;
                }
            }

            // Check if troughs are similar (within 0.5%)
            if (firstTrough < double.MaxValue && secondTrough < double.MaxValue)
            {
                var diff = Math.Abs(firstTrough - secondTrough) / firstTrough;
                if (diff < 0.005)
                {
                    // Check for peak between troughs
                    var peak = 0.0;
                    for (int i = firstTroughIdx + 1; i < currentIndex; i++)
                    {
                        peak = Math.Max(peak, (double)bars[i].High);
                    }

                    if (peak > firstTrough * 1.02) // Peak at least 2% above troughs
                        return 1.0;
                }
            }

            return 0;
        }

        private double DetectHeadShoulders(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 30) return 0;

            // Simplified head and shoulders detection
            // Look for three peaks with middle one highest
            var peaks = new List<(int index, double height)>();

            for (int i = currentIndex - 29; i <= currentIndex; i++)
            {
                if (i < 2 || i >= currentIndex - 1) continue;

                if (bars[i].High > bars[i - 1].High && bars[i].High > bars[i + 1].High &&
                    bars[i].High > bars[i - 2].High && bars[i].High > bars[i + 2].High)
                {
                    peaks.Add((i, (double)bars[i].High));
                }
            }

            if (peaks.Count >= 3)
            {
                // Check if middle peak is highest (head)
                var sorted = peaks.OrderByDescending(p => p.height).ToList();
                if (sorted[0].index > peaks[0].index && sorted[0].index < peaks[peaks.Count - 1].index)
                {
                    // Check if shoulders are approximately equal
                    var leftShoulder = peaks.First().height;
                    var rightShoulder = peaks.Last().height;
                    if (Math.Abs(leftShoulder - rightShoulder) / leftShoulder < 0.02)
                        return 1.0;
                }
            }

            return 0;
        }

        private double DetectInverseHeadShoulders(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 30) return 0;

            // Simplified inverse head and shoulders detection
            // Look for three troughs with middle one lowest
            var troughs = new List<(int index, double depth)>();

            for (int i = currentIndex - 29; i <= currentIndex; i++)
            {
                if (i < 2 || i >= currentIndex - 1) continue;

                if (bars[i].Low < bars[i - 1].Low && bars[i].Low < bars[i + 1].Low &&
                    bars[i].Low < bars[i - 2].Low && bars[i].Low < bars[i + 2].Low)
                {
                    troughs.Add((i, (double)bars[i].Low));
                }
            }

            if (troughs.Count >= 3)
            {
                // Check if middle trough is lowest (head)
                var sorted = troughs.OrderBy(t => t.depth).ToList();
                if (sorted[0].index > troughs[0].index && sorted[0].index < troughs[troughs.Count - 1].index)
                {
                    // Check if shoulders are approximately equal
                    var leftShoulder = troughs.First().depth;
                    var rightShoulder = troughs.Last().depth;
                    if (Math.Abs(leftShoulder - rightShoulder) / leftShoulder < 0.02)
                        return 1.0;
                }
            }

            return 0;
        }

        private double DetectTriangle(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return 0;

            // Detect converging price action
            var highs = new List<double>();
            var lows = new List<double>();

            for (int i = currentIndex - 19; i <= currentIndex; i++)
            {
                highs.Add((double)bars[i].High);
                lows.Add((double)bars[i].Low);
            }

            // Calculate trend of highs and lows
            var highTrend = CalculateTrend(highs);
            var lowTrend = CalculateTrend(lows);

            // Triangle if highs trending down and lows trending up (converging)
            if (highTrend < -0.0001 && lowTrend > 0.0001)
                return 1.0;

            return 0;
        }

        private double DetectFlag(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 15) return 0;

            // Flag: strong move followed by consolidation
            var initialMove = (double)(bars[currentIndex - 14].Close - bars[currentIndex - 15].Close);

            if (Math.Abs(initialMove) > (double)bars[currentIndex - 15].Close * 0.002) // 0.2% move
            {
                // Check for consolidation in recent bars
                var recentRange = 0.0;
                for (int i = currentIndex - 10; i <= currentIndex; i++)
                {
                    recentRange = Math.Max(recentRange, (double)(bars[i].High - bars[i].Low));
                }

                if (recentRange < Math.Abs(initialMove) * 0.5) // Consolidation range less than half of initial move
                    return initialMove > 0 ? 1.0 : -1.0;
            }

            return 0;
        }

        private double DetectWedge(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return 0;

            // Wedge: converging trend lines with both moving in same direction
            var highs = new List<double>();
            var lows = new List<double>();

            for (int i = currentIndex - 19; i <= currentIndex; i++)
            {
                highs.Add((double)bars[i].High);
                lows.Add((double)bars[i].Low);
            }

            var highTrend = CalculateTrend(highs);
            var lowTrend = CalculateTrend(lows);

            // Rising wedge (bearish) or falling wedge (bullish)
            if (highTrend > 0 && lowTrend > 0 && highTrend < lowTrend)
                return -1.0; // Rising wedge (bearish)
            else if (highTrend < 0 && lowTrend < 0 && highTrend > lowTrend)
                return 1.0; // Falling wedge (bullish)

            return 0;
        }

        private double CalculateTrend(List<double> values)
        {
            var n = values.Count;
            if (n < 2) return 0;

            var sumX = 0.0;
            var sumY = 0.0;
            var sumXY = 0.0;
            var sumX2 = 0.0;

            for (int i = 0; i < n; i++)
            {
                sumX += i;
                sumY += values[i];
                sumXY += i * values[i];
                sumX2 += i * i;
            }

            return (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        }

        private double CalculatePatternSuccessRate(IReadOnlyList<OhlcBar> bars, int currentIndex, int patternCount)
        {
            if (patternCount == 0 || currentIndex < 10) return 0.5;

            // Look back at recent patterns and check if they were followed by expected moves
            int successCount = 0;
            int totalPatterns = 0;

            for (int i = Math.Max(10, currentIndex - 100); i < currentIndex - 5; i++)
            {
                // Simplified: check if there was any pattern activity
                if (bars[i].TickVolume > bars[i - 1].TickVolume * 1.1)
                {
                    totalPatterns++;
                    // Check if next 5 bars moved in expected direction
                    var nextMove = (double)(bars[i + 5].Close - bars[i].Close);
                    if (Math.Abs(nextMove) > 0.0001)
                        successCount++;
                }
            }

            return totalPatterns > 0 ? (double)successCount / totalPatterns : 0.5;
        }

        public override void Reset()
        {
            _patternHistory.Clear();
            _swingHighs.Clear();
            _swingLows.Clear();
        }
    }

}
