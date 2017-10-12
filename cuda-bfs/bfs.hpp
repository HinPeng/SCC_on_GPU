#include <vector>
#include <string>

using namespace std;

const int N = 14;
const int M = 19;

void Graphpreproc(const string filename, vector<unsigned> &VF, vector<unsigned> &EF, vector<unsigned> &VB, vector<unsigned> &EB);
void BFS(vector<unsigned> &V, vector<unsigned> &E, unsigned sourceVertex, std::vector<unsigned> & distances);
