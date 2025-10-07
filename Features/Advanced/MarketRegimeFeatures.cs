using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Base;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.Advanced
{
    public class MarketRegimeFeatures : BaseFeatureCalculator
    {
        public override string Name => "MarketRegime";
        public override string Category => "Regime";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(5);
        public override int Priority => 13;

        private readonly RollingWindow<RegimeSnapshot> _regimeHistory = new(100);

        public class RegimeSnapshot
        {
            public DateTime Timestamp { get; set; }
            public double Volatility { get; set; }
            public double Trend { get; set; }
            public double Momentum { get; set; }
            public int RegimeType { get; set; } // 0=Range, 1=Trend, 2=Volatile
        }

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 50) return;

            // Detect current regime
            var (regimeType, confidence) = DetectRegime(bars, currentIndex);
            output.AddFeature("fg2_regime_type", regimeType);
            output.AddFeature("fg2_regime_confidence", confidence);

            // Calculate regime duration
            if (_regimeHistory.Count > 0)
            {
                int duration = 1;
                var currentRegime = regimeType;

                foreach (var snapshot in _regimeHistory.GetValues())
                {
                    if (snapshot.RegimeType == currentRegime)
                        duration++;
                    else
                        break;
                }
                output.AddFeature("fg2_regime_duration", duration);
            }
            else
            {
                output.AddFeature("fg2_regime_duration", 1);
            }

            // Transition probability
            if (_regimeHistory.Count >= 50)
            {
                var transitionProb = CalculateTransitionProbability();
                output.AddFeature("fg2_regime_transition_prob", transitionProb);
            }
            else
            {
                output.AddFeature("fg2_regime_transition_prob", 0.0);
            }

            // GARCH volatility estimation
            var garchVol = EstimateGARCHVolatility(bars, currentIndex);
            output.AddFeature("fg2_garch_volatility", garchVol);

            // Jump detection
            var jumpIntensity = DetectJumps(bars, currentIndex);
            output.AddFeature("fg2_jump_intensity", jumpIntensity);

            // Vol of vol
            var volOfVol = CalculateVolOfVol(bars, currentIndex);
            output.AddFeature("fg2_vol_of_vol", volOfVol);

            // Fractal dimension
            var fractalDim = CalculateFractalDimension(bars, currentIndex);
            output.AddFeature("fg2_fractal_dimension", fractalDim);

            // Detrended fluctuation analysis
            var dfa = CalculateDFA(bars, currentIndex);
            output.AddFeature("fg2_detrended_fluctuation", dfa);

            // Market efficiency
            var efficiencyRatio = CalculateEfficiencyRatio(bars, currentIndex);
            output.AddFeature("fg2_efficiency_ratio", efficiencyRatio);

            // Variance ratio test
            var varianceRatio = CalculateVarianceRatio(bars, currentIndex);
            output.AddFeature("fg2_variance_ratio", varianceRatio);

            // Tail risk metrics
            CalculateTailRisk(output, bars, currentIndex);

            // Update history
            _regimeHistory.Add(new RegimeSnapshot
            {
                Timestamp = bars[currentIndex].Timestamp,
                Volatility = garchVol,
                Trend = efficiencyRatio,
                Momentum = 0,
                RegimeType = (int)regimeType
            });
        }

        private (double type, double confidence) DetectRegime(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Calculate metrics for regime detection
            var volatility = ATR(bars, 14, currentIndex);
            var avgVolatility = 0.0;

            for (int i = currentIndex - 49; i <= currentIndex; i++)
            {
                avgVolatility += ATR(bars, 14, i);
            }
            avgVolatility /= 50;

            // Trend strength
            var ema20 = EMA(bars, 20, currentIndex);
            var ema50 = EMA(bars, 50, currentIndex);
            var trendStrength = Math.Abs(SafeDiv(ema20 - ema50, ema50));

            // Determine regime
            double regimeType = 0; // Range-bound
            double confidence = 0.5;

            if (volatility > avgVolatility * 1.5)
            {
                regimeType = 2; // Volatile
                confidence = SafeDiv(volatility, avgVolatility * 2);
            }
            else if (trendStrength > 0.001)
            {
                regimeType = 1; // Trending
                confidence = Math.Min(1.0, trendStrength * 1000);
            }

            return (regimeType, Math.Min(1.0, confidence));
        }

        private double CalculateTransitionProbability()
        {
            var regimes = _regimeHistory.GetValues().Take(50).Select(r => r.RegimeType).ToList();
            int transitions = 0;

            for (int i = 1; i < regimes.Count; i++)
            {
                if (regimes[i] != regimes[i - 1])
                    transitions++;
            }

            return (double)transitions / regimes.Count;
        }

        private double EstimateGARCHVolatility(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Simplified GARCH(1,1) estimation
            const double omega = 0.000001;
            const double alpha = 0.05;
            const double beta = 0.94;

            var returns = new List<double>();
            for (int i = currentIndex - 29; i <= currentIndex; i++)
            {
                if (i > 0)
                {
                    var ret = Math.Log((double)bars[i].Close / (double)bars[i - 1].Close);
                    returns.Add(ret);
                }
            }

            var unconditionalVar = returns.Select(r => r * r).Average();
            double garchVar = unconditionalVar;

            foreach (var ret in returns)
            {
                garchVar = omega + alpha * ret * ret + beta * garchVar;
            }

            return Math.Sqrt(garchVar);
        }

        private double DetectJumps(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Detect price jumps using bipower variation
            double bipowerVar = 0;
            int jumps = 0;

            for (int i = currentIndex - 29; i <= currentIndex; i++)
            {
                if (i > 1)
                {
                    var ret1 = Math.Log((double)bars[i].Close / (double)bars[i - 1].Close);
                    var ret2 = Math.Log((double)bars[i - 1].Close / (double)bars[i - 2].Close);
                    bipowerVar += Math.Abs(ret1) * Math.Abs(ret2);

                    // Jump threshold (3 standard deviations)
                    if (Math.Abs(ret1) > 3 * Math.Sqrt(bipowerVar / (i - currentIndex + 30)))
                        jumps++;
                }
            }

            return (double)jumps / 30;
        }

        private double CalculateVolOfVol(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var volatilities = new List<double>();

            for (int i = currentIndex - 29; i <= currentIndex; i++)
            {
                if (i >= 14)
                {
                    volatilities.Add(ATR(bars, 14, i));
                }
            }

            if (volatilities.Count < 2) return 0;

            var mean = volatilities.Average();
            var variance = volatilities.Select(v => Math.Pow(v - mean, 2)).Average();

            return Math.Sqrt(variance) / mean;
        }

        private double CalculateFractalDimension(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Box-counting method simplified
            var prices = new List<double>();
            for (int i = currentIndex - 49; i <= currentIndex; i++)
            {
                prices.Add((double)bars[i].Close);
            }

            var maxPrice = prices.Max();
            var minPrice = prices.Min();
            var range = maxPrice - minPrice;

            if (range < 1e-10) return 1.5;

            // Count boxes at different scales
            int[] boxSizes = { 2, 4, 8, 16 };
            var boxCounts = new List<double>();

            foreach (var size in boxSizes)
            {
                var boxes = new HashSet<(int, int)>();
                var step = prices.Count / size;

                for (int i = 0; i < prices.Count - 1; i++)
                {
                    var x1 = i / step;
                    var y1 = (int)((prices[i] - minPrice) / range * size);
                    var x2 = (i + 1) / step;
                    var y2 = (int)((prices[i + 1] - minPrice) / range * size);

                    boxes.Add((x1, y1));
                    boxes.Add((x2, y2));
                }

                boxCounts.Add(boxes.Count);
            }

            // Calculate fractal dimension using log-log regression
            var logSizes = boxSizes.Select(s => Math.Log(s)).ToArray();
            var logCounts = boxCounts.Select(c => Math.Log(c)).ToArray();

            var n = logSizes.Length;
            var sumX = logSizes.Sum();
            var sumY = logCounts.Sum();
            var sumXY = logSizes.Zip(logCounts, (x, y) => x * y).Sum();
            var sumX2 = logSizes.Select(x => x * x).Sum();

            var slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);

            return Math.Abs(slope);
        }

        private double CalculateDFA(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Detrended Fluctuation Analysis
            var prices = new List<double>();
            for (int i = currentIndex - 49; i <= currentIndex; i++)
            {
                prices.Add((double)bars[i].Close);
            }

            // Calculate cumulative sum
            var mean = prices.Average();
            var cumSum = new double[prices.Count];
            cumSum[0] = prices[0] - mean;

            for (int i = 1; i < prices.Count; i++)
            {
                cumSum[i] = cumSum[i - 1] + prices[i] - mean;
            }

            // Calculate fluctuation for different box sizes
            int[] boxSizes = { 4, 8, 16 };
            var fluctuations = new List<double>();

            foreach (var boxSize in boxSizes)
            {
                var nBoxes = cumSum.Length / boxSize;
                double sumF2 = 0;

                for (int v = 0; v < nBoxes; v++)
                {
                    var segment = cumSum.Skip(v * boxSize).Take(boxSize).ToArray();

                    // Fit linear trend
                    var xValues = Enumerable.Range(0, boxSize).Select(x => (double)x).ToArray();
                    var n = boxSize;
                    var sumX = xValues.Sum();
                    var sumY = segment.Sum();
                    var sumXY = xValues.Zip(segment, (x, y) => x * y).Sum();
                    var sumX2 = xValues.Select(x => x * x).Sum();

                    var slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
                    var intercept = (sumY - slope * sumX) / n;

                    // Calculate fluctuation
                    for (int i = 0; i < boxSize; i++)
                    {
                        var trend = slope * i + intercept;
                        var diff = segment[i] - trend;
                        sumF2 += diff * diff;
                    }
                }

                fluctuations.Add(Math.Sqrt(sumF2 / (nBoxes * boxSize)));
            }

            // Calculate scaling exponent (simplified)
            return fluctuations.Average() / prices.Count;
        }

        private double CalculateEfficiencyRatio(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return 0.5;

            // Kaufman's Efficiency Ratio
            var change = Math.Abs((double)(bars[currentIndex].Close - bars[currentIndex - 19].Close));
            double volatility = 0;

            for (int i = currentIndex - 18; i <= currentIndex; i++)
            {
                volatility += Math.Abs((double)(bars[i].Close - bars[i - 1].Close));
            }

            return SafeDiv(change, volatility, 0.5);
        }

        private double CalculateVarianceRatio(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Lo-MacKinlay variance ratio test
            var returns = new List<double>();
            for (int i = currentIndex - 39; i <= currentIndex; i++)
            {
                if (i > 0)
                {
                    returns.Add(Math.Log((double)bars[i].Close / (double)bars[i - 1].Close));
                }
            }

            // Calculate variances at different frequencies
            var var1 = returns.Select(r => r * r).Average();

            var returns2 = new List<double>();
            for (int i = 1; i < returns.Count; i += 2)
            {
                if (i > 0)
                {
                    returns2.Add(returns[i] + returns[i - 1]);
                }
            }

            var var2 = returns2.Select(r => r * r).Average() / 2;

            return SafeDiv(var2, var1, 1.0);
        }

        private void CalculateTailRisk(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var returns = new List<double>();
            for (int i = currentIndex - 99; i <= currentIndex; i++)
            {
                if (i > 0)
                {
                    returns.Add(Math.Log((double)bars[i].Close / (double)bars[i - 1].Close));
                }
            }

            returns.Sort();

            // Calculate VaR at 5% and 95%
            var leftTail = returns.Take((int)(returns.Count * 0.05)).Average();
            var rightTail = returns.Skip((int)(returns.Count * 0.95)).Average();

            output.AddFeature("avd_left_tail_risk", Math.Abs(leftTail));
            output.AddFeature("avd_right_tail_risk", Math.Abs(rightTail));
            output.AddFeature("avd_tail_asymmetry", SafeDiv(Math.Abs(rightTail), Math.Abs(leftTail), 1.0));
        }

        public override void Reset()
        {
            _regimeHistory.Clear();
        }
    }
}
