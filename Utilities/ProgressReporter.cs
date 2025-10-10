using System.Diagnostics;

namespace ForexFeatureGenerator.Utilities
{
    public class ProgressReporter
    {
        private readonly string _taskName;
        private readonly int _totalItems;
        private readonly Stopwatch _stopwatch;
        private int _lastReportedPercentage;
        private readonly int _reportInterval;

        public ProgressReporter(string taskName, int totalItems, int reportInterval = 10)
        {
            _taskName = taskName;
            _totalItems = totalItems;
            _reportInterval = reportInterval;
            _stopwatch = Stopwatch.StartNew();
            _lastReportedPercentage = 0;

            Console.WriteLine($"  Starting: {_taskName} ({totalItems:N0} items)");
        }

        public void Update(int currentItem)
        {
            var percentage = (int)((currentItem + 1) * 100.0 / _totalItems);

            if (percentage >= _lastReportedPercentage + _reportInterval || percentage == 100)
            {
                _lastReportedPercentage = percentage;

                var elapsed = _stopwatch.Elapsed;
                var itemsProcessed = currentItem + 1;
                var itemsPerSecond = itemsProcessed / elapsed.TotalSeconds;
                var estimatedTotal = TimeSpan.FromSeconds(_totalItems / itemsPerSecond);
                var estimatedRemaining = estimatedTotal - elapsed;

                Console.WriteLine($"    {percentage}% complete - " +
                                $"{itemsProcessed:N0}/{_totalItems:N0} items - " +
                                $"{itemsPerSecond:F0} items/sec - " +
                                $"ETA: {FormatTimeSpan(estimatedRemaining)}");
            }
        }

        public void Complete()
        {
            _stopwatch.Stop();
            var totalTime = _stopwatch.Elapsed;
            var itemsPerSecond = _totalItems / totalTime.TotalSeconds;

            Console.WriteLine($"  ✓ Completed: {_taskName}");
            Console.WriteLine($"    Total time: {FormatTimeSpan(totalTime)} - Average: {itemsPerSecond:F1} items/sec");
        }

        private string FormatTimeSpan(TimeSpan ts)
        {
            if (ts.TotalSeconds < 0)
                return "calculating...";

            if (ts.TotalMinutes < 1)
                return $"{ts.Seconds}s";
            else if (ts.TotalHours < 1)
                return $"{ts.Minutes}m {ts.Seconds}s";
            else
                return $"{(int)ts.TotalHours}h {ts.Minutes}m";
        }
    }
}