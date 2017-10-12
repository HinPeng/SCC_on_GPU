#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>

using namespace std;

const int N = 5;
const int M = 6;
const string fn("test.txt");

void Graphpreproc(const string filename, vector<int> &a, vector<int> &b);

int main()
{
	vector<int> a, b;
	vector<int>::iterator itr;
	a.reverse(N+1);
	b.reverse(M);
	Graphpreproc(fn, a, b);
	for (itr = a.begin(); itr != a.end(); itr++)
		cout << *itr << ' ';
	cout << endl;
	for (itr = b.begin(); itr != b.end(); itr++)
		cout << *itr << ' ';
	cout << endl;
	system("PAUSE");
	return 0;
}

void Graphpreproc(const string filename, vector<int> &a, vector<int> &b)
{
	ifstream in_f;
	vector<int> t;
	vector<int>::iterator itr;
	int count = 0;
	in_f.open(filename, ios::in);
	while (!in_f.eof()){
		string temp, s1, s2;
		stringstream ss1, ss2;
		int t1, t2;
		getline(in_f, temp);
		if (*(temp.begin()) == '#')
			continue;
		s1 = string(temp,0, temp.find_first_of('\t'));
		s2 = string(temp,temp.find_first_not_of('\t', temp.find_first_of('\t')), temp.find_last_not_of('\t'));
		ss1 << s1;
		ss1 >> t1;
		ss2 << s2;
		ss2 >> t2;
		t.push_back(t1);
		b.push_back(t2);
	}

	itr = t.begin();
	if (*itr == 0)
		a.push_back(0);
	else
		a.push_back(-1);
	for (int i = 0; i < N - 1; i++){
		while ((*itr == i)&&(itr != t.end())){
			count++;
			itr++;
		}
		count += a.at(i);
		a.push_back(count);
		count = 0;
	}
	a.push_back(M);
}
