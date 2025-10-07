namespace ForexFeatureGenerator.Core.Models
{
    public record TickData
    {
        public DateTime Timestamp { get; init; }
        public decimal Bid { get; init; }
        public decimal Ask { get; init; }

        public decimal MidPrice => (Bid + Ask) / 2;
        public decimal Spread => Ask - Bid;
    }
}
