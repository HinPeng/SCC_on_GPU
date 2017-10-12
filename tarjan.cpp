#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <stack>
#include <time.h>

using namespace std;

const int N = 2394385;
const int M = 5021410;

const string filename("WikiTalk.txt");

static int tarjan_count;
static int tarjan_dfn[N];
static int tarjan_low[N];
static bool tarjan_instack[N];
stack<int> tarjan_stack;
vector<int> V;
vector<int> E;
vector<int> SCC;

int com_num;

void init() {
    V.reserve(N+1);
    E.reserve(M);
        
    tarjan_count = -1;
    com_num = 0;
    memset(tarjan_dfn, -1, sizeof(tarjan_dfn));
    memset(tarjan_low, -1, sizeof(tarjan_low));
        
    memset(tarjan_instack, false, sizeof(tarjan_instack));
    }
void Graphpreproc(const string filename) {
    ifstream in_f;
    vector<unsigned> t;
    vector<unsigned>::iterator itr;
        
    unsigned count = 0;
    in_f.open(filename.c_str(), ios::in);
    while (!in_f.eof()){
        string temp, s1, s2;
        stringstream ss1, ss2;
        unsigned t1, t2;
        getline(in_f, temp);
        if(temp.length() == 0)
            continue;
        if (*(temp.begin()) == '#')
            continue;
        s1 = string(temp, 0, temp.find_first_of('\t'));
        s2 = string(temp, temp.find_first_not_of('\t', temp.find_first_of('\t')), temp.find_last_not_of('\t'));
        ss1 << s1;
        ss1 >> t1;
        ss2 << s2;
        ss2 >> t2;
        t.push_back(t1);
        E.push_back(t2);
    }
    
    itr = t.begin();
    V.push_back(0);
    for (int i = 0; i < N - 1; i++){
        while ((itr != t.end()) && (*itr == i)){
            count++;
            itr++;
        }
        count += V.at(i);
        V.push_back(count);
        count = 0;
    }
    V.push_back(M);
}
    
void Tarjan(int v) {        // Need TarjanInit()
    tarjan_count++;
    tarjan_dfn[v] = tarjan_count;
    tarjan_low[v] = tarjan_count;
        
    tarjan_stack.push(v);
    tarjan_instack[v] = true;
        
    for (int edge = V[v]; edge < V[v + 1]; ++edge) {
        int neighbor = E[edge];
        if (tarjan_dfn[neighbor] == -1) {
            Tarjan(neighbor);
            tarjan_low[v] = min(tarjan_low[v], tarjan_low[neighbor]);
        }
        else if (tarjan_instack[neighbor]) {
            tarjan_low[v] = min(tarjan_low[v], tarjan_dfn[neighbor]);
        }
    }
        
    if (tarjan_dfn[v] == tarjan_low[v]) {
        com_num++;
        int i = 0;
        while (tarjan_stack.top() != v) {
            tarjan_instack[tarjan_stack.top()] = false;
            i++;
            tarjan_stack.pop();
        }
        tarjan_instack[tarjan_stack.top()] = false;
        tarjan_stack.pop();
        SCC.push_back(++i);
    }
}
static bool compare(const int &a, const int &b) {
    return a > b;
}
    
void show(){
    sort(SCC.begin(),SCC.end(),compare);
    if(com_num > 20)
        com_num = 20;
    for(int i = 0;i<com_num;i++)
        cout<<SCC.at(i)<<endl;
}


int main() {
    clock_t start1=clock();
    init();
    Graphpreproc(filename);
     
    clock_t start=clock();
    for (int i = 0; i < N; i++) {
        if (tarjan_dfn[i] == -1) {
            Tarjan(i);
        }
    }
    clock_t finish=clock();
    printf("time elapsed:%.2fs\n",(double)(finish-start)/1.0e6);
    show();
    clock_t finish1=clock();
    printf("time elapsed:%.2fs\n",(double)(finish1-start1)/1.0e6);
    
    return 0;
}

