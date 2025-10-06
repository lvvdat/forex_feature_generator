namespace ForexFeatureGenerator.Core.Infrastructure
{
    public class RollingWindow<T>
    {
        private readonly T[] _buffer;
        private readonly int _size;
        private int _head; // Points to most recent item
        private int _count;

        public RollingWindow(int size)
        {
            _size = size;
            _buffer = new T[size];
            _head = -1;
            _count = 0;
        }

        public void Add(T item)
        {
            _head = (_head + 1) % _size;
            _buffer[_head] = item;
            if (_count < _size) _count++;
        }

        // [0] = most recent, [1] = second most recent, etc.
        public T this[int i]
        {
            get
            {
                if (i < 0 || i >= _count)
                    throw new IndexOutOfRangeException($"Index {i} out of range [0, {_count})");

                int idx = (_head - i + _size) % _size;
                return _buffer[idx];
            }
        }

        public int Count => _count;
        public bool IsFull => _count == _size;

        public IEnumerable<T> GetValues()
        {
            for (int i = 0; i < _count; i++)
                yield return this[i];
        }

        public void Clear()
        {
            Array.Clear(_buffer);
            _count = 0;
            _head = -1;
        }
    }
}
