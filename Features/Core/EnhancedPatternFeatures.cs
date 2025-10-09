using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ForexFeatureGenerator.Features.Pattern
{
    /// <summary>
    /// Enhanced pattern recognition for price action
    /// </summary>
    public class EnhancedPatternFeatures : BaseCalculator
    {
        public override string Name => "EnhancedPattern";
        public override string Category => "Pattern";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(1);
        public override int Priority => 5;

        private readonly RollingWindow<int> _patternHistory = new(50);

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return;

            var patterns = DetectPatterns(bars, currentIndex);

            // === 1. Pattern Strength ===
            // Combined pattern signal
            var bullishCount = patterns.Count(p => p.Value > 0);
            var bearishCount = patterns.Count(p => p.Value < 0);

            var patternStrength = SafeDiv(bullishCount - bearishCount,
                                         bullishCount + bearishCount + 1);
            output.AddFeature("pattern_strength", patternStrength);

            // === 2. Pattern Confidence ===
            // How many patterns agree
            var patternConfidence = 0.0;
            if (bullishCount > bearishCount && bearishCount == 0)
                patternConfidence = 1.0;
            else if (bearishCount > bullishCount && bullishCount == 0)
                patternConfidence = -1.0;

            output.AddFeature("pattern_confidence", patternConfidence);

            // === 3. Key Patterns ===
            // Only include most important patterns
            output.AddFeature("engulfing", patterns.GetValueOrDefault("engulfing", 0));
            output.AddFeature("pin_bar_pattern", patterns.GetValueOrDefault("pin_bar", 0));
            output.AddFeature("inside_bar", patterns.GetValueOrDefault("inside_bar", 0));
            output.AddFeature("breakout", patterns.GetValueOrDefault("breakout", 0));

            // === 4. Pattern Frequency ===
            _patternHistory.Add(bullishCount - bearishCount);

            if (_patternHistory.Count >= 20)
            {
                var recentPatterns = _patternHistory.GetValues().Take(20).Average();
                output.AddFeature("pattern_frequency", Sigmoid(recentPatterns));
            }

            // === 5. Support/Resistance Test ===
            var srTest = DetectSupportResistanceTest(bars, currentIndex);
            output.AddFeature("sr_test", srTest);

            // === 6. Trend Continuation/Reversal ===
            var trendPattern = DetectTrendPattern(bars, currentIndex);
            output.AddFeature("trend_pattern", trendPattern);
        }

        private Dictionary<string, double> DetectPatterns(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var patterns = new Dictionary<string, double>();

            if (currentIndex < 2) return patterns;

            var curr = bars[currentIndex];
            var prev1 = bars[currentIndex - 1];
            var prev2 = bars[currentIndex - 2];

            // Current bar metrics
            var currBody = Math.Abs((double)(curr.Close - curr.Open));
            var currRange = (double)(curr.High - curr.Low);
            var currUpper = (double)(curr.High - Math.Max(curr.Open, curr.Close));
            var currLower = (double)(Math.Min(curr.Open, curr.Close) - curr.Low);

            // Previous bar metrics
            var prevBody = Math.Abs((double)(prev1.Close - prev1.Open));
            var prevRange = (double)(prev1.High - prev1.Low);

            // === Engulfing Pattern ===
            if (prevBody > 0 && currBody > 0)
            {
                // Bullish engulfing
                if (prev1.Close < prev1.Open && curr.Close > curr.Open &&
                    curr.Open <= prev1.Close && curr.Close >= prev1.Open)
                {
                    patterns["engulfing"] = 1.0;
                }
                // Bearish engulfing
                else if (prev1.Close > prev1.Open && curr.Close < curr.Open &&
                        curr.Open >= prev1.Close && curr.Close <= prev1.Open)
                {
                    patterns["engulfing"] = -1.0;
                }
                else
                {
                    patterns["engulfing"] = 0.0;
                }
            }

            // === Pin Bar ===
            if (currRange > 0)
            {
                var bodyRatio = currBody / currRange;

                // Bullish pin bar (long lower wick)
                if (bodyRatio < 0.3 && currLower > currUpper * 2)
                {
                    patterns["pin_bar"] = 1.0;
                }
                // Bearish pin bar (long upper wick)
                else if (bodyRatio < 0.3 && currUpper > currLower * 2)
                {
                    patterns["pin_bar"] = -1.0;
                }
                else
                {
                    patterns["pin_bar"] = 0.0;
                }
            }

            // === Inside Bar ===
            if (curr.High <= prev1.High && curr.Low >= prev1.Low)
            {
                patterns["inside_bar"] = 1.0; // Consolidation
            }
            else
            {
                patterns["inside_bar"] = 0.0;
            }

            // === Breakout Pattern ===
            if (currentIndex >= 20)
            {
                var high20 = double.MinValue;
                var low20 = double.MaxValue;

                for (int i = currentIndex - 19; i <= currentIndex - 1; i++)
                {
                    high20 = Math.Max(high20, (double)bars[i].High);
                    low20 = Math.Min(low20, (double)bars[i].Low);
                }

                if ((double)curr.Close > high20)
                    patterns["breakout"] = 1.0;
                else if ((double)curr.Close < low20)
                    patterns["breakout"] = -1.0;
                else
                    patterns["breakout"] = 0.0;
            }

            return patterns;
        }

        private double DetectSupportResistanceTest(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 50) return 0;

            // Find recent swing points
            var swingHighs = new List<double>();
            var swingLows = new List<double>();

            for (int i = currentIndex - 48; i <= currentIndex - 2; i++)
            {
                if (i < 2) continue;

                var bar = bars[i];
                var prev = bars[i - 1];
                var next = bars[i + 1];

                // Swing high
                if (bar.High > prev.High && bar.High > next.High)
                {
                    swingHighs.Add((double)bar.High);
                }

                // Swing low
                if (bar.Low < prev.Low && bar.Low < next.Low)
                {
                    swingLows.Add((double)bar.Low);
                }
            }

            var close = (double)bars[currentIndex].Close;

            // Test resistance
            foreach (var resistance in swingHighs)
            {
                if (Math.Abs(close - resistance) / resistance < 0.001)
                    return -1.0; // Testing resistance
            }

            // Test support
            foreach (var support in swingLows)
            {
                if (Math.Abs(close - support) / support < 0.001)
                    return 1.0; // Testing support
            }

            return 0;
        }

        private double DetectTrendPattern(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 10) return 0;

            // Higher highs and higher lows (uptrend)
            var highs = new List<double>();
            var lows = new List<double>();

            for (int i = currentIndex - 9; i <= currentIndex; i++)
            {
                highs.Add((double)bars[i].High);
                lows.Add((double)bars[i].Low);
            }

            // Check trend structure
            var highsIncreasing = true;
            var lowsIncreasing = true;
            var highsDecreasing = true;
            var lowsDecreasing = true;

            for (int i = 1; i < highs.Count; i++)
            {
                if (highs[i] <= highs[i - 1]) highsIncreasing = false;
                if (lows[i] <= lows[i - 1]) lowsIncreasing = false;
                if (highs[i] >= highs[i - 1]) highsDecreasing = false;
                if (lows[i] >= lows[i - 1]) lowsDecreasing = false;
            }

            if (highsIncreasing && lowsIncreasing)
                return 1.0; // Strong uptrend
            else if (highsDecreasing && lowsDecreasing)
                return -1.0; // Strong downtrend
            else
                return 0; // No clear trend
        }

        public override void Reset()
        {
            base.Reset();
            _patternHistory.Clear();
        }
    }
}