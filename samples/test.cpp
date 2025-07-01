#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using timer = std::chrono::steady_clock;

void by_value(vector<int> v)
{
    vector<int> copy = v;  // Copy constructor
}

void by_ref(vector<int>& v)
{
    vector<int> copy = v;  // 
}

void by_const_ref(const vector<int>& v)
{
    vector<int> copy = v;  // Still a copy (v is const but we're copying)
}

void by_move(vector<int>&& v)
{
    vector<int> moved = std::move(v);  // Move constructor
}

void by_move2(vector<int>& v)
{
    vector<int> moved = std::move(v);  // Move constructor
}

int main()
{
    const size_t N = 10'000'000; // 10 milyon eleman
    vector<int> original(N, 42); // hepsi 42 olan vektör

    cout << "Benchmark:\n";
    
    auto t1 = timer::now();
    by_value(original);         // Kopyalanýr
    auto t2 = timer::now();
    cout << "by value\n" << chrono::duration_cast<chrono::microseconds>(t2 - t1).count()
        << " us\n";

    t1 = timer::now();
    by_ref(original);         // Kopyalanýr
    t2 = timer::now();
    cout << "by ref\n" << chrono::duration_cast<chrono::microseconds>(t2 - t1).count()
        << " us\n";

    t1 = timer::now();
    by_const_ref(original);     // Yine kopyalanýr (copy constructor)
    t2 = timer::now();
    cout << "by const ref\n" << chrono::duration_cast<chrono::microseconds>(t2 - t1).count()
        << " us\n";

    t1 = timer::now();
    by_move(std::move(original)); // Taþýnýr (move constructor)
    t2 = timer::now();
    cout << "by move\n" << chrono::duration_cast<chrono::microseconds>(t2 - t1).count()
        << " us\n";

    vector<int> original2(N, 42); // hepsi 42 olan vektör
    
    t1 = timer::now();
    by_move2(original2);
    t2 = timer::now();
    cout << "by move 2\n" << chrono::duration_cast<chrono::microseconds>(t2 - t1).count()
        << " us\n";

    return 0;
}
