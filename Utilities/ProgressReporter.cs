using System.Diagnostics;

namespace ForexFeatureGenerator.Utilities
{
    public class ProgressReporter
    {
        private readonly int _total;
        private readonly string _title;
        private readonly Stopwatch _stopwatch;

        public ProgressReporter(string title, int total)
        {
            _title = title;
            _total = total;
            _stopwatch = Stopwatch.StartNew();
        }

        public void Update(int current, string extra = "")
        {
            var percent = (double)current / _total;
            var bar = new string('█', (int)(percent * 30)) + new string('░', 30 - (int)(percent * 30));
            Console.Write($"\r  {_title}: [{bar}] {percent:P0} {extra}");
        }

        public void Complete()
        {
            Update(_total);
            Console.WriteLine($" ✓ ({_stopwatch.Elapsed:mm\\:ss})");
        }
    }
}
