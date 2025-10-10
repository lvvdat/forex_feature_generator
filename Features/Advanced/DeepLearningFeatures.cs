using ForexFeatureGenerator.Core.Models;
using ForexFeatureGenerator.Features.Core;
using ForexFeatureGenerator.Core.Infrastructure;

namespace ForexFeatureGenerator.Features.Advanced
{
    /// <summary>
    /// Deep Learning optimized features including sequence representations,
    /// attention-like mechanisms, embeddings, and temporal patterns
    /// </summary>
    public class DeepLearningFeatures : BaseFeatureCalculator
    {
        public override string Name => "DeepLearning";
        public override string Category => "DL_Optimized";
        public override TimeSpan Timeframe => TimeSpan.FromMinutes(5);
        public override int Priority => 6;

        private readonly RollingWindow<AttentionSnapshot> _attentionHistory = new(50);

        // For temporal convolution-like features
        private readonly int[] _kernelSizes = { 3, 5, 7, 9 };

        public class AttentionSnapshot
        {
            public DateTime Timestamp { get; set; }
            public double[]? AttentionWeights { get; set; }
            public double FocusScore { get; set; }
        }

        public override void Calculate(FeatureVector output, IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            if (currentIndex < 20) return;

            var bar = bars[currentIndex];

            // ===== SEQUENCE EMBEDDINGS =====
            // Create price sequence embeddings (like word embeddings for time series)
            var priceSeq = ExtractPriceSequence(bars, currentIndex, 10);
            var priceEmbedding = CalculateSequenceEmbedding(priceSeq);
            output.AddFeature("06_dl_price_embedding", priceEmbedding);

            // ===== ATTENTION-LIKE MECHANISMS =====
            // Calculate attention weights for recent history
            var attentionWeights = CalculateAttentionWeights(bars, currentIndex, 20);
            var contextVector = CalculateContextVector(bars, currentIndex, attentionWeights);

            output.AddFeature("06_dl_attention_spread", CalculateAttentionSpread(attentionWeights));
            output.AddFeature("06_dl_context_strength", contextVector);

            // Store attention snapshot
            _attentionHistory.Add(new AttentionSnapshot
            {
                Timestamp = bar.Timestamp,
                AttentionWeights = attentionWeights,
                FocusScore = attentionWeights.Max()
            });

            // ===== TEMPORAL CONVOLUTION FEATURES =====
            // Simulate 1D convolutions over time series
            foreach (var kernelSize in _kernelSizes)
            {
                if (currentIndex >= kernelSize)
                {
                    var convOutput = ApplyTemporalConvolution(bars, currentIndex, kernelSize);
                    output.AddFeature($"06_dl_conv_{kernelSize}_price", convOutput.priceConv);
                }
            }

            // ===== POOLING OPERATIONS =====
            // Max pooling
            var (maxPoolPrice, maxPoolVolume) = ApplyMaxPooling(bars, currentIndex, 20, 5);
            output.AddFeature("06_dl_maxpool_price", maxPoolPrice);

            // Average pooling
            var (avgPoolPrice, avgPoolVolume) = ApplyAveragePooling(bars, currentIndex, 20, 5);
            output.AddFeature("06_dl_avgpool_price", avgPoolPrice);

            // ===== RECURRENT FEATURES (LSTM-like) =====

            // Gates simulation
            var (forgetGate, inputGate, outputGate) = CalculatePseudoGates(bars, currentIndex);
            output.AddFeature("06_dl_forget_gate", forgetGate);
            output.AddFeature("06_dl_input_gate", inputGate);

            // ===== TEMPORAL PATTERNS =====
            // Identify recurring patterns in sequences
            if (currentIndex >= 30)
            {
                var patternScore = DetectRecurringPattern(bars, currentIndex);
                output.AddFeature("06_dl_pattern_score", patternScore);

                var cycleStrength = DetectCyclicBehavior(bars, currentIndex);
                output.AddFeature("06_dl_cycle_strength", cycleStrength);
            }

            // ===== AUTOENCODER-LIKE FEATURES =====
            // Dimensionality reduction representation
            var bottleneck = CalculateBottleneckRepresentation(bars, currentIndex);
            output.AddFeature("06_dl_bottleneck_feat", bottleneck);

            // ===== MULTI-SCALE TEMPORAL FEATURES =====
            // Features at different time resolutions
            if (currentIndex >= 50)
            {
                var scales = new[] { 5, 10, 20, 50 };
                foreach (var scale in scales)
                {
                    var multiScaleFeat = CalculateMultiScaleFeature(bars, currentIndex, scale);
                    output.AddFeature($"06_dl_multiscale_{scale}", multiScaleFeat);
                }
            }

            // ===== SEQUENCE-TO-SEQUENCE FEATURES =====
            // Encoder-decoder style features
            var encodedSeq = EncodeSequence(bars, currentIndex, 20);
            output.AddFeature("06_dl_encoded_seq", encodedSeq);

            var decodedState = DecodeToCurrentState(bars, currentIndex);
            output.AddFeature("06_dl_decoded_state", decodedState);

            // ===== GRAPH-LIKE FEATURES =====
            // Time series as graph nodes with edges
            var nodeImportance = CalculateNodeImportance(bars, currentIndex);
            output.AddFeature("06_dl_node_importance", nodeImportance);

            // ===== TRANSFORMER-LIKE FEATURES =====
            // Positional encoding
            var posEncoding = CalculatePositionalEncoding(currentIndex);
            output.AddFeature("06_dl_pos_encoding_sin", posEncoding.sin);
            output.AddFeature("06_dl_pos_encoding_cos", posEncoding.cos);

            // ===== LAYER NORMALIZATION =====
            // Normalized features across the sequence
            var layerNorm = CalculateLayerNormalization(bars, currentIndex);
            output.AddFeature("06_dl_layer_norm", layerNorm);

            // ===== DROPOUT-LIKE FEATURES =====
            // Feature importance through random dropping
            var robustness = CalculateFeatureRobustness(bars, currentIndex);
            output.AddFeature("06_dl_feature_robustness", robustness);

            // ===== SEQUENCE COMPLEXITY =====
            // Measure of sequence predictability
            if (currentIndex >= 30)
            {
                var complexity = CalculateSequenceComplexity(bars, currentIndex);
                output.AddFeature("06_dl_sequence_complexity", complexity);

                var entropy = CalculateSequenceEntropy(bars, currentIndex);
                output.AddFeature("06_dl_sequence_entropy", entropy);
            }
        }

        private double[] ExtractPriceSequence(IReadOnlyList<OhlcBar> bars, int currentIndex, int length)
        {
            var sequence = new double[length];
            for (int i = 0; i < length; i++)
            {
                sequence[i] = (double)bars[currentIndex - length + 1 + i].Close;
            }
            return sequence;
        }

        private double CalculateSequenceEmbedding(double[] sequence)
        {
            // Simple embedding using statistical moments
            var mean = sequence.Average();
            var std = Math.Sqrt(sequence.Select(v => Math.Pow(v - mean, 2)).Average());
            var skewness = CalculateSkewness(sequence);

            // Combine into single embedding value
            return (mean * 0.5 + std * 0.3 + skewness * 0.2);
        }

        private double CalculateSkewness(double[] values)
        {
            if (values.Length < 3) return 0;

            var mean = values.Average();
            var std = Math.Sqrt(values.Select(v => Math.Pow(v - mean, 2)).Average());
            if (std < 1e-10) return 0;

            var n = values.Length;
            var sum = values.Sum(v => Math.Pow((v - mean) / std, 3));

            return sum * n / ((n - 1) * (n - 2));
        }

        private double[] CalculateAttentionWeights(IReadOnlyList<OhlcBar> bars, int currentIndex, int lookback)
        {
            var weights = new double[lookback];
            var currentPrice = (double)bars[currentIndex].Close;

            // Calculate similarity scores (attention)
            for (int i = 0; i < lookback; i++)
            {
                var pastPrice = (double)bars[currentIndex - lookback + 1 + i].Close;
                var similarity = Math.Exp(-Math.Pow(currentPrice - pastPrice, 2) / (2 * 0.01)); // Gaussian kernel
                weights[i] = similarity;
            }

            // Softmax normalization
            var maxWeight = weights.Max();
            var expWeights = weights.Select(w => Math.Exp(w - maxWeight)).ToArray();
            var sumExp = expWeights.Sum();

            for (int i = 0; i < lookback; i++)
            {
                weights[i] = expWeights[i] / sumExp;
            }

            return weights;
        }

        private double CalculateContextVector(IReadOnlyList<OhlcBar> bars, int currentIndex, double[] weights)
        {
            double context = 0;
            for (int i = 0; i < weights.Length; i++)
            {
                var price = (double)bars[currentIndex - weights.Length + 1 + i].Close;
                context += weights[i] * price;
            }
            return context;
        }

        private double CalculateAttentionSpread(double[] weights)
        {
            // Measure how spread out the attention is (entropy)
            double entropy = 0;
            foreach (var w in weights)
            {
                if (w > 1e-10)
                {
                    entropy -= w * Math.Log(w);
                }
            }
            return entropy;
        }

        private (double priceConv, double volumeConv) 
            ApplyTemporalConvolution(IReadOnlyList<OhlcBar> bars, int currentIndex, int kernelSize)
        {
            // Simple learned kernel (averaging with exponential decay)
            double priceSum = 0;
            double volumeSum = 0;
            double weightSum = 0;

            for (int i = 0; i < kernelSize; i++)
            {
                var weight = Math.Exp(-i * 0.1); // Exponential decay
                priceSum += weight * (double)bars[currentIndex - i].Close;
                volumeSum += weight * bars[currentIndex - i].TickVolume;
                weightSum += weight;
            }

            return (priceSum / weightSum, volumeSum / weightSum);
        }

        private (double price, double volume) 
            ApplyMaxPooling(IReadOnlyList<OhlcBar> bars, int currentIndex, int windowSize, int poolSize)
        {
            var maxPrice = double.MinValue;
            var maxVolume = double.MinValue;

            for (int i = currentIndex - windowSize + 1; i <= currentIndex; i += poolSize)
            {
                var price = (double)bars[i].High; // Use high for max pooling
                var volume = (double)bars[i].TickVolume;

                maxPrice = Math.Max(maxPrice, price);
                maxVolume = Math.Max(maxVolume, volume);
            }

            return (maxPrice, maxVolume);
        }

        private (double price, double volume) 
            ApplyAveragePooling(IReadOnlyList<OhlcBar> bars, int currentIndex, int windowSize, int poolSize)
        {
            var prices = new List<double>();
            var volumes = new List<double>();

            for (int i = currentIndex - windowSize + 1; i <= currentIndex; i += poolSize)
            {
                prices.Add((double)bars[i].Close);
                volumes.Add(bars[i].TickVolume);
            }

            return (prices.Average(), volumes.Average());
        }

        private (double forget, double input, double output) 
            CalculatePseudoGates(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Simulate LSTM gates using volatility and momentum
            var atr = CalculateATR(bars, currentIndex, 14);
            var avgAtr = 0.0;
            for (int i = currentIndex - 19; i <= currentIndex; i++)
            {
                avgAtr += CalculateATR(bars, i, 14);
            }
            avgAtr /= 20;

            // Forget gate: high when volatility is high (forget old patterns)
            var forgetGate = Math.Min(1.0, atr / avgAtr);

            // Input gate: high when new information is significant
            var priceChange = Math.Abs((double)(bars[currentIndex].Close - bars[currentIndex - 1].Close));
            var inputGate = Math.Min(1.0, priceChange / (atr + 1e-10));

            // Output gate: based on trend strength
            var ema9 = CalculateEMA(bars, currentIndex,9);
            var ema21 = CalculateEMA(bars, currentIndex, 21);
            var outputGate = Math.Abs(SafeDiv(ema9 - ema21, ema21));

            return (forgetGate, inputGate, outputGate);
        }

        private double DetectRecurringPattern(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Autocorrelation at various lags
            var lags = new[] { 5, 10, 15, 20 };
            var maxCorr = 0.0;

            foreach (var lag in lags)
            {
                if (currentIndex >= lag * 2)
                {
                    var corr = CalculateAutocorrelation(bars, currentIndex, lag);
                    maxCorr = Math.Max(maxCorr, Math.Abs(corr));
                }
            }

            return maxCorr;
        }

        private double CalculateAutocorrelation(IReadOnlyList<OhlcBar> bars, int currentIndex, int lag)
        {
            var values = new List<double>();
            for (int i = currentIndex - 2 * lag; i <= currentIndex; i++)
            {
                values.Add((double)bars[i].Close);
            }

            var mean = values.Average();
            double numerator = 0;
            double denominator = 0;

            for (int i = lag; i < values.Count; i++)
            {
                numerator += (values[i] - mean) * (values[i - lag] - mean);
            }

            for (int i = 0; i < values.Count; i++)
            {
                denominator += Math.Pow(values[i] - mean, 2);
            }

            return denominator > 0 ? numerator / denominator : 0;
        }

        private double DetectCyclicBehavior(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Detect cycles using FFT approximation (peak detection in frequency domain)
            var prices = new List<double>();
            for (int i = currentIndex - 29; i <= currentIndex; i++)
            {
                prices.Add((double)bars[i].Close);
            }

            // Simplified cycle detection using autocorrelation peaks
            var maxCycle = 0.0;
            for (int period = 3; period <= 10; period++)
            {
                var corr = CalculateAutocorrelation(bars, currentIndex, period);
                maxCycle = Math.Max(maxCycle, corr);
            }

            return maxCycle;
        }

        private double CalculateBottleneckRepresentation(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // PCA-like compression to single dimension
            var prices = new List<double>();
            var volumes = new List<double>();

            for (int i = currentIndex - 19; i <= currentIndex; i++)
            {
                prices.Add((double)bars[i].Close);
                volumes.Add(bars[i].TickVolume);
            }

            // First principal component (simplified)
            var priceMean = prices.Average();
            var volumeMean = volumes.Average();

            var covariance = 0.0;
            for (int i = 0; i < prices.Count; i++)
            {
                covariance += (prices[i] - priceMean) * (volumes[i] - volumeMean);
            }
            covariance /= prices.Count;

            return covariance;
        }

        private double CalculateMultiScaleFeature(IReadOnlyList<OhlcBar> bars, int currentIndex, int scale)
        {
            // Moving average at different scales
            return CalculateSMA(bars, currentIndex, scale);
        }

        private double EncodeSequence(IReadOnlyList<OhlcBar> bars, int currentIndex, int length)
        {
            // Encode sequence to single value (like seq2seq encoder)
            var sequence = ExtractPriceSequence(bars, currentIndex, length);
            return CalculateSequenceEmbedding(sequence);
        }

        private double DecodeToCurrentState(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Decode from latent representation
            return (double)bars[currentIndex].Close;
        }

        private double CalculateNodeImportance(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Graph node importance (like PageRank)
            var currentVolume = (double)bars[currentIndex].TickVolume;
            var avgVolume = 0.0;

            for (int i = currentIndex - 9; i <= currentIndex; i++)
            {
                avgVolume += bars[i].TickVolume;
            }
            avgVolume /= 10;

            return SafeDiv(currentVolume, avgVolume);
        }

        private (double sin, double cos) CalculatePositionalEncoding(int position)
        {
            // Transformer-style positional encoding
            var dimension = 64;
            var angle = position / Math.Pow(10000, 2.0 / dimension);

            return (Math.Sin(angle), Math.Cos(angle));
        }

        private double CalculateLayerNormalization(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var prices = new List<double>();
            for (int i = currentIndex - 19; i <= currentIndex; i++)
            {
                prices.Add((double)bars[i].Close);
            }

            var mean = prices.Average();
            var std = Math.Sqrt(prices.Select(p => Math.Pow(p - mean, 2)).Average());

            var current = (double)bars[currentIndex].Close;
            return SafeDiv(current - mean, std + 1e-10);
        }

        private double CalculateFeatureRobustness(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Robustness = consistency across different window sizes
            var ema5 = CalculateEMA(bars, currentIndex, 5);
            var ema10 = CalculateEMA(bars, currentIndex, 10);
            var ema20 = CalculateEMA(bars, currentIndex, 20);

            var variance = new[] { ema5, ema10, ema20 }.Select(v => Math.Pow(v - new[] { ema5, ema10, ema20 }.Average(), 2)).Average();

            return 1.0 / (1.0 + variance);
        }

        private double CalculateSequenceComplexity(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            // Lempel-Ziv complexity approximation
            var sequence = ExtractPriceSequence(bars, currentIndex, 30);

            // Discretize
            var mean = sequence.Average();
            var binary = sequence.Select(v => v > mean ? 1 : 0).ToArray();

            // Count unique patterns
            var patterns = new HashSet<string>();
            for (int len = 1; len <= 5; len++)
            {
                for (int i = 0; i <= binary.Length - len; i++)
                {
                    var pattern = string.Join("", binary.Skip(i).Take(len));
                    patterns.Add(pattern);
                }
            }

            return (double)patterns.Count / 30;
        }

        private double CalculateSequenceEntropy(IReadOnlyList<OhlcBar> bars, int currentIndex)
        {
            var sequence = ExtractPriceSequence(bars, currentIndex, 30);

            // Discretize into bins
            var bins = 10;
            var min = sequence.Min();
            var max = sequence.Max();
            var binWidth = (max - min) / bins;

            if (binWidth < 1e-10) return 0;

            var counts = new int[bins];
            foreach (var value in sequence)
            {
                var bin = (int)((value - min) / binWidth);
                if (bin >= bins) bin = bins - 1;
                counts[bin]++;
            }

            double entropy = 0;
            foreach (var count in counts)
            {
                if (count > 0)
                {
                    var p = (double)count / sequence.Length;
                    entropy -= p * Math.Log(p, 2);
                }
            }

            return entropy;
        }

        public override void Reset()
        {
            _attentionHistory.Clear();
        }
    }
}