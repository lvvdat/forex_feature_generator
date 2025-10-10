namespace ForexFeatureGenerator.Core.Models
{
    public enum NormalizationType
    {
        None,              // Keep as is
        StandardScaler,    // Z-score normalization
        RobustScaler,      // Using median and IQR
        QuantileTransform, // Uniform distribution [0, 1]
        MinMaxScaler      // Scale to [-1, 1]
    }

    public class NormalizationConfig
    {
        private readonly Dictionary<string, NormalizationType> _featureNormalization;

        public NormalizationConfig()
        {
            _featureNormalization = new Dictionary<string, NormalizationType>();
            InitializeConfiguration();
        }

        private void InitializeConfiguration()
        {
            // Features that don't need normalization (already in [-1, 1] or [0, 1])
            var keepAsIs = new[]
            {
                "01_dir_candle_direction", "01_dir_mean_reversion_prob", "01_dir_momentum_accel",
                "01_dir_price_position", "01_dir_trend_efficiency", "01_dir_vol_mom_correlation",
                "01_dir_volume_direction", "01_dir_volume_pressure",

                "02_cyclical_phase", "02_market_stress", "02_regime_directional_bias",
                "02_regime_duration_norm", "02_regime_momentum", "02_trend_efficiency",
                "02_trend_mtf_alignment", "02_trend_mtf_strength", "02_vol_trend",

                "03_micro_buy_pressure", "03_micro_flow_acceleration", "03_micro_flow_imbalance",
                "03_micro_pressure_diff", "03_micro_price_efficiency", "03_micro_sell_pressure",
                "03_micro_spike_direction", "03_micro_spread_zscore", "03_micro_tick_direction",
                "03_micro_volume_spike", "03_micro_vwap_deviation",

                "04_tech_bb_expansion", "04_tech_bb_squeeze", "04_tech_ma_alignment",
                "04_tech_ma_convergence", "04_tech_ma_dev_21", "04_tech_ma_dev_9",
                "04_tech_macd_normalized", "04_tech_macd_quality", "04_tech_rsi_composite",
                "04_tech_rsi_normalized", "04_tech_vol_percentile",

                "05_pos_long_entry_score", "05_pos_long_trailing_active", "05_pos_mtf_consensus",
                "05_pos_mtf_long_alignment", "05_pos_mtf_short_alignment", "05_pos_resistance_strength",
                "05_pos_short_entry_score", "05_pos_short_trailing_active", "05_pos_support_strength",

                "06_dl_input_gate", "06_dl_pos_encoding_cos", "06_dl_pos_encoding_sin",

                "07_ml_hour_american", "07_ml_hour_asian", "07_ml_hour_european",
                "07_ml_price_percentile_50", "07_ml_price_volume_correlation", "07_ml_volume_percentile_50"
            };

            // StandardScaler for symmetric continuous features
            var standardScalerCols = new[]
            {
                "01_dir_pattern_strength", "03_micro_depth_imbalance", "04_tech_bb_position",
                "05_pos_long_max_favorable", "05_pos_short_max_favorable", "05_pos_stop_distance",
                "06_dl_layer_norm", "07_ml_composite_trend_score", "07_ml_composite_volatility_score"
            };

            // RobustScaler for features with outliers
            var robustScalerCols = new[]
            {
                "01_dir_dm_minus", "01_dir_dm_plus", "01_dir_momentum_z10", "01_dir_momentum_z5",
                "01_dir_trend_strength",

                "02_hurst_exponent", "02_market_condition_score", "02_regime_confidence", "02_trend_quality",

                "03_micro_spread_volume_ratio", "03_micro_tick_clustering",

                "04_tech_atr_ratio", "04_tech_rsi_momentum",

                "05_pos_distance_to_long_entry", "05_pos_distance_to_short_entry", "05_pos_downside_risk",
                "05_pos_expected_long_duration", "05_pos_expected_short_duration", "05_pos_long_profit_potential",
                "05_pos_long_risk_reward", "05_pos_risk_asymmetry", "05_pos_short_profit_potential",
                "05_pos_short_risk_reward", "05_pos_upside_potential",

                "06_dl_attention_spread", "06_dl_bottleneck_feat", "06_dl_cycle_strength",
                "06_dl_encoded_seq", "06_dl_feature_robustness", "06_dl_forget_gate",
                "06_dl_node_importance", "06_dl_pattern_score", "06_dl_price_embedding",
                "06_dl_sequence_entropy",

                "07_ml_composite_momentum_score", "07_ml_market_quality_score", "07_ml_pct_change_lag_5",
                "07_ml_price_entropy", "07_ml_price_volume_covariance", "07_ml_price_zscore",
                "07_ml_returns_cubed", "07_ml_returns_squared", "07_ml_volume_entropy",
                "07_ml_volume_gini", "07_ml_volume_short_long_ratio", "07_ml_volume_zscore"
            };

            // QuantileTransformer for skewed distributions
            var quantileTransformerCols = new[]
            {
                "05_pos_optimal_long_entry", "05_pos_optimal_short_entry",
                "06_dl_avgpool_price", "06_dl_context_strength", "06_dl_conv_3_price",
                "06_dl_conv_5_price", "06_dl_conv_7_price", "06_dl_conv_9_price",
                "06_dl_decoded_state", "06_dl_maxpool_price", "06_dl_multiscale_10",
                "06_dl_multiscale_20", "06_dl_multiscale_5", "06_dl_multiscale_50",
                "07_ml_price_lag_1", "07_ml_price_lag_3", "07_ml_price_lag_5"
            };

            // MinMaxScaler for discrete features
            var minMaxScalerCols = new[]
            {
                "01_dir_momentum_quality", "02_regime_stability", "02_regime_type",
                "03_micro_amihud_illiquidity", "03_micro_iceberg_pattern", "03_micro_stop_hunt",
                "03_micro_tick_intensity", "05_pos_long_expectancy", "05_pos_long_quality",
                "05_pos_long_success_prob", "05_pos_recommended_size_long", "05_pos_recommended_size_short",
                "05_pos_short_expectancy", "05_pos_short_quality", "05_pos_short_success_prob",
                "05_pos_size_confidence", "06_dl_sequence_complexity", "07_ml_price_bin", "07_ml_volume_bin"
            };

            // Set normalization types
            foreach (var feature in keepAsIs)
                _featureNormalization[feature] = NormalizationType.None;

            foreach (var feature in standardScalerCols)
                _featureNormalization[feature] = NormalizationType.StandardScaler;

            foreach (var feature in robustScalerCols)
                _featureNormalization[feature] = NormalizationType.RobustScaler;

            foreach (var feature in quantileTransformerCols)
                _featureNormalization[feature] = NormalizationType.QuantileTransform;

            foreach (var feature in minMaxScalerCols)
                _featureNormalization[feature] = NormalizationType.MinMaxScaler;
        }

        public NormalizationType GetNormalizationType(string featureName)
        {
            return _featureNormalization.TryGetValue(featureName, out var type)
                ? type
                : NormalizationType.StandardScaler; // Default
        }

        public HashSet<string> GetFeaturesByType(NormalizationType type)
        {
            return _featureNormalization
                .Where(kvp => kvp.Value == type)
                .Select(kvp => kvp.Key)
                .ToHashSet();
        }
    }
}