#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <windows.h>
using namespace std;
#define MAX_DEPTH 10

static int depth = 0;
// 决策树:constructor
struct tree 
{
    string attribute;             // 节点的属性
    vector<tree*> branches;       // 子节点列表
    vector<string> labels;
    string classification;        // 叶子节点的分类结果
    // 构造函数，初始化节点
    tree(const string& attr) : attribute(attr), classification("") {}
};

// 决策树: selector :判断是否是叶子
bool is_leaf(tree* node)
{
    return node->classification != "";
}

// constructor of Sample: implement data abstraction
struct Sample 
{
    vector<string> attributes;       // 属性集合
    string classification;           // 分类结果
};

tree* buildDecisionTree(const vector<Sample>& samples, vector<string>& attributes, char flag);
double predictData(tree* dicisionTree, vector<Sample> testSet,bool flag);
void destroyDecisionTree(tree* node);
/*****************************************************************************************/


/*****************************************************************************************/
// selector of Sample : data_abstraction:得到数据类型Sample的classification,即分类结果
string get_class(Sample sample)
{
    return sample.classification;
}
// 拿到samples中的占大多数的classification
string get_majority_class(const vector<Sample>&samples)
{
    unordered_map<string, vector<Sample>> counter;
    for (size_t i = 0; i < samples.size(); i++) {
        counter[samples[i].classification].push_back(samples[i]);
    }
    size_t sum = 0;
    string getclass;
    for (const auto& sample : counter) {
        if (sample.second.size() > sum) {
            getclass = sample.first;
            sum = sample.second.size();
        }
    }
    return getclass;
}

// 计算信息熵
double calEntropy(const vector<Sample>& samples) 
{
    unordered_set<string> uniqueClasses;  //创建了一个无序的集合,用于存储classification,unordered_set中的元素是不重复的.
    for (const Sample& sample : samples) { 
        uniqueClasses.insert(sample.classification); //容器的方法:uniqueClasses.insert()
    }
    double entropy = 0.0;
    for (const string& uniqueClass : uniqueClasses) {
        size_t count = 0;
        for (const Sample& s : samples) {
            if (s.classification == uniqueClass) {
                count++;
            }
        }
        double probability = double(count) / samples.size();
        entropy -= probability * log2(probability); //熵的计算公式
    }
    return entropy;
}

// 计算信息增益
double calGain(const vector<Sample>& samples, const string& attribute,size_t index)
{
    double initialEntropy = calEntropy(samples);

    unordered_map<string, vector<Sample>> splitSamples; //创建了类似于字典的映射:键:string-->值:vector<Sample>
    for (const Sample& sample : samples) {
        splitSamples[sample.attributes[index]].push_back(sample);
    }

    for (const auto& sample : splitSamples) {
        double temp = calEntropy(sample.second);
        double part = 1.0 * sample.second.size() / samples.size();
        initialEntropy -= part * temp;
    }
    return initialEntropy;
}

// 选择最佳分裂属性
string chooseBestAttribute(const vector<Sample>& samples, const vector<string>& attributes)
{
    double gain_ent = -1;
    string choose_string=attributes[0];
    double temp = calGain(samples, attributes[0],0);
    for (size_t i = 1; i < attributes.size(); i++) {
        if ((temp = calGain(samples, attributes[i], i)) > gain_ent) {
            choose_string = attributes[i];
            gain_ent = temp;
        }
    }
    return choose_string;
}
// 判断是否是同一属性.
bool sameClass(const vector<Sample>& samples)
{
    if (samples.empty())
        return true;
    string myclass = samples[0].classification;
    for (size_t i = 1; i < samples.size(); i++) {
        if (samples[i].classification != myclass)
            return false;
    }
    return true;
}
// 判断是否应该停止.
bool stop(const vector<Sample>& samples, const vector<string>& attributes)
{
    return attributes.empty() || sameClass(samples) || depth == MAX_DEPTH;
}
//将samples根据attribute分类
unordered_map<string, vector< Sample >> splitSamples(const vector<Sample>& samples,size_t index)
{
    unordered_map<string, vector<Sample>> list_sample;
    for (const auto& sample : samples) {
        list_sample[sample.attributes[index]].push_back(sample);
    }
    
    return list_sample;
}

// 判断是否进行预剪枝
bool judge_prepruning(const tree* node, const vector<Sample>&samples, size_t index)
{
    double test1 = 0.0;
    double test2 = 0.0;

    //case 1:进行预剪枝
    tree* node1 = new tree(*node);
    tree* temp = new tree("");
    temp->classification = get_majority_class(samples);
    node1->branches.push_back(temp);
    test1 = predictData(node1, samples,false);

    //case 2:不进行预剪枝
    unordered_map<string, vector<Sample>> listsample = splitSamples(samples, index);
    tree* node2 = new tree(*node);
    for (const auto& sample : listsample) {
        node2->labels.push_back(sample.first);
        tree* leaf = new tree("");
        leaf->classification = get_majority_class(sample.second);
        node2->branches.push_back(leaf);
    }
    test2 = predictData(node2, samples,false);

    destroyDecisionTree(node1);
    destroyDecisionTree(node2);

    return test1 >= test2;

}
// 构建决策树
tree* buildDecisionTree(const vector<Sample>& samples, vector<string>& attributes, bool flag)
{
    tree* node = new tree("");  // 创建一个新节点
    if (stop(samples, attributes)) {
        //如果达到结束的条件即可结束递归
        node->classification = get_majority_class(samples);
        return node;
    }
    else {
        string label = chooseBestAttribute(samples, attributes);
        node->attribute = label;
        //在attributes中删除label
        size_t index;            //index为attributes中label的位置
        for (index = 0; index < attributes.size(); index++) {
            if (label == attributes[index]) {
                attributes.erase(attributes.begin() + index);
                break;
            }
        }
        // 不允许预剪枝
        if (flag == false) {
            unordered_map<string, vector<Sample>> listsample = splitSamples(samples, index);//返回一个以label为分类准则的字典
            depth++;
            for (const auto& sample : listsample) {
                node->branches.push_back(buildDecisionTree(sample.second, attributes, flag));
                node->labels.push_back(sample.first); // labels为非叶子节点下方的树枝的方向
            }
        }
        // 允许预剪枝
        else {
            // case 1: 不进行预剪枝
            if (!judge_prepruning(node, samples, index)) {
                unordered_map<string, vector<Sample>> listsample = splitSamples(samples,index);//返回一个以label为分类准则的字典
                depth++;
                for (const auto& sample : listsample) {
                    node->branches.push_back(buildDecisionTree(sample.second, attributes, flag));
                    node->labels.push_back(sample.first); // labels为非叶子节点下方的树枝的方向
                }
            }
            // case 2: 进行预剪枝
            else {
                tree* add_node = new tree("");
                add_node->classification = get_majority_class(samples);
                node->branches.push_back(add_node);
            }
        }
        return node;
    }
}
// 打印决策树
void printTree(tree* node, int spaces = 0)
{
    if (is_leaf(node)) {
        return;
    }
    cout << setw(spaces) << " " << node->attribute << endl;
    for (size_t i = 0; i < node->branches.size(); i++) {
        printTree(node->branches[i], spaces + 2);
    }

}

//销毁决策树
void destroyDecisionTree(tree* node)
{
    if (node != nullptr) {
        for (size_t i = 0; i < node->branches.size(); i++) {
            destroyDecisionTree(node->branches[i]);
        }
        delete node;
    }
}
//利用决策树预测
string predictTree(tree* decisionTree, vector<string> test)
{
    if (is_leaf(decisionTree)) {
        return decisionTree->classification;
    }
    else {
        size_t index = 0;
        bool success = false;
        for (string word : test) {
            success = false;
            for (index = 0; index < decisionTree->labels.size(); index++) {
                if (word == decisionTree->labels[index]) {
                    success = true;
                    break;
                }
            }
            if (success)
                break;
        }
        if (success)
            return predictTree(decisionTree->branches[index], test);
        else 
            return decisionTree->branches[0]->classification;
    }
}
double predictData(tree* dicisionTree, vector<Sample> testSet,bool flag)
{
    int judge = 0;

    for (size_t i = 0; i < testSet.size(); i++) {
        if(flag)
            cout << predictTree(dicisionTree, testSet[i].attributes) << endl;
        if (predictTree(dicisionTree, testSet[i].attributes) == testSet[i].classification) {
            judge++;
        }
    }
    return 1.0 * judge / testSet.size();
}

void input(vector<Sample>& set) 
{
    while (true) {
        Sample temp;
        set.push_back(temp);
        for (int j = 0; j < 9; j++) {
            string str;
            cin >> str;
            if (str.empty()) {
                break;
            }
            if (j < 8)
                set.back().attributes.push_back(str);
            else
                set.back().classification = str;
        }
        if (cin.eof())
            break;
    }
}
// 主函数
int main()
{
    if (0) {
        vector<Sample> trainingData = {
        {{"青绿","蜷缩","浊响","清晰","凹陷","硬滑"},"是"},
        {{"乌黑","蜷缩","沉闷","清晰","凹陷","硬滑"},"是"},
        {{"乌黑","蜷缩","浊响","清晰","凹陷","硬滑"},"是"},
        {{"青绿","稍蜷","浊响","清晰","稍凹","软粘"},"是"},
        {{"乌黑","稍蜷","浊响","稍糊","稍凹","软粘"},"是"},
        {{"青绿","硬挺","清脆","清晰","平坦","软粘"},"否"},
        {{"浅白","稍蜷","沉闷","稍糊","凹陷","硬滑"},"否"},
        {{"乌黑","稍蜷","浊响","清晰","稍凹","软粘"},"否"},
        {{"浅白","蜷缩","浊响","模糊","平坦","硬滑"},"否"},
        {{"青绿","蜷缩","沉闷","稍糊","稍凹","硬滑"},"否"},
        };
        vector<Sample> testData = {
        {{"青绿","蜷缩","沉闷","清晰","凹陷","硬滑"},"是"},
        {{"浅白","蜷缩","浊响","清晰","凹陷","硬滑"},"是"},
        {{"乌黑","稍蜷","浊响","清晰","稍凹","硬滑"},"是"},
        {{"乌黑","稍蜷","沉闷","稍糊","稍凹","硬滑"},"是"},
        {{"浅白","硬挺","清脆","模糊","平坦","硬滑"},"否"},
        {{"浅白","蜷缩","浊响","模糊","平坦","软粘"},"否"},
        {{"青绿","稍蜷","浊响","稍糊","凹陷","硬滑"},"否"},
        //add more
        };

        vector<string> attributes = { "色泽","根蒂","敲声","纹理","脐部","触感" };
        cout << "case 1:" << endl;
        tree* decisionTree = buildDecisionTree(trainingData, attributes, true);   //构造决策树
        printTree(decisionTree);   //打印
        bool flag1 = true;
        bool flag2 = true;
        if (flag1 == true) {
            cout << "训练集重新预测结果" << endl;
        }
        double test1 = predictData(decisionTree, trainingData, flag1);
        if (flag2 == true) {
            cout << "测试集预测结果" << endl;
        }
        double test2 = predictData(decisionTree, testData, flag2);
        cout << "训练集的准确度:" << test1 << endl;
        cout << "测试集的准确度:" << test2 << endl;

        destroyDecisionTree(decisionTree);
        cout << "样例一结束" << endl;
    }
    cout << "<=========================================================>" << endl;
    depth = 0;
    if (1) {
        //输入训练集
        vector<Sample> trainingData;
        FILE* file = freopen("nursery_trainingdata.txt", "r", stdin);
        if (file == nullptr) {
            cerr << "Failed to open nursery_trainingdata.txt for reading!请将nursery_trainingdata置于该cpp文件的同一目录下" << endl;
            return 1; // 退出程序，返回错误码
        }
        input(trainingData);

        vector <Sample> testData = {
        {{"usual","very_crit","foster","more","less_conv","convenient","nonprob","priority"},"spec_prior"},
        {{"usual","very_crit","foster","more","less_conv","convenient","nonprob","not_recom"},"not_recom"},
        {{"usual","very_crit","foster","more","less_conv","convenient","slightly_prob","recommended"},"spec_prior"},
        {{"usual","very_crit","foster","more","less_conv","convenient","slightly_prob","priority"},"spec_prior"},
        {{"usual","very_crit","foster","more","less_conv","convenient","slightly_prob","not_recom"},"not_recom"},
        {{"usual","very_crit","foster","more","less_conv","convenient","problematic","recommended"},"spec_prior"},
        {{"usual","very_crit","foster","more","less_conv","convenient","problematic","priority"},"spec_prior"},
        {{"usual","very_crit","foster","more","less_conv","convenient","problematic","not_recom"},"not_recom"},
        {{"usual","very_crit","foster","more","less_conv","inconv","nonprob","recommended"},"spec_prior"},
        {{"usual","very_crit","foster","more","less_conv","inconv","nonprob","priority"},"spec_prior"},
        {{"usual","very_crit","foster","more","less_conv","inconv","nonprob","not_recom"},"not_recom"},
        {{"usual","very_crit","foster","more","less_conv","inconv","slightly_prob","recommended"},"spec_prior"},
        {{"usual","very_crit","foster","more","less_conv","inconv","slightly_prob","priority"},"spec_prior"},
        {{"usual","very_crit","foster","more","less_conv","inconv","slightly_prob","not_recom"},"not_recom"},
        {{"usual","very_crit","foster","more","less_conv","inconv","problematic","recommended"},"spec_prior"},
        {{"usual","very_crit","foster","more","less_conv","inconv","problematic","priority"},"spec_prior"},
        {{"usual","very_crit","foster","more","less_conv","inconv","problematic","not_recom"},"not_recom"},
        {{"usual","very_crit","foster","more","critical","convenient","nonprob","recommended"},"spec_prior"},
        {{"usual","very_crit","foster","more","critical","convenient","nonprob","priority"},"spec_prior"},
        {{"usual","very_crit","foster","more","critical","convenient","nonprob","not_recom"},"not_recom"},
        {{"usual","very_crit","foster","more","critical","inconv","slightly_prob","recommended"},"spec_prior"},
        {{"usual","very_crit","foster","more","critical","inconv","slightly_prob","priority"},"spec_prior"},
        {{"usual","very_crit","foster","more","critical","inconv","slightly_prob","not_recom"},"not_recom"},
        {{"usual","very_crit","foster","more","critical","inconv","problematic","recommended"},"spec_prior"},
        {{"usual","very_crit","foster","more","critical","inconv","problematic","priority"},"spec_prior"},
        {{"usual","very_crit","foster","more","critical","inconv","problematic","not_recom"},"not_recom"},
        {{"pretentious","proper","complete","2","convenient","convenient","nonprob","priority"},"priority"},
        {{"pretentious","proper","complete","2","convenient","convenient","nonprob","not_recom"},"not_recom"},
        {{"pretentious","proper","complete","2","convenient","convenient","slightly_prob","recommended"},"very_recom"},
        {{"pretentious","proper","complete","2","convenient","convenient","slightly_prob","priority"},"priority"},
        {{"pretentious","proper","complete","2","convenient","convenient","slightly_prob","not_recom"},"not_recom"},
        {{"pretentious","proper","complete","2","convenient","convenient","problematic","recommended"},"priority"},
        {{"pretentious","proper","complete","2","convenient","convenient","problematic","priority"},"priority"},
        {{"pretentious","proper","complete","2","convenient","convenient","problematic","not_recom"},"not_recom"},
        {{"pretentious","proper","complete","2","convenient","inconv","nonprob","recommended"},"very_recom"},
        {{"pretentious","proper","complete","2","convenient","inconv","nonprob","priority"},"priority"},
        {{"great_pret","improper","incomplete","3","critical","inconv","slightly_prob","priority"},"spec_prior"},
        {{"great_pret","improper","incomplete","3","critical","inconv","slightly_prob","not_recom"},"not_recom"},
        {{"great_pret","improper","incomplete","3","critical","inconv","problematic","recommended"},"spec_prior"},
        {{"great_pret","improper","incomplete","3","critical","inconv","problematic","priority"},"spec_prior"},
        {{"great_pret","improper","igreat_pret","improper","incomplete","3","critical","inconv"},"slightly_prob"},
        {{"recommended","spec_prior","ncomplete","3","critical","inconv","problematic","not_recom"},"not_recom"},
        {{"great_pret","improper","incomplete","more","convenient","convenient","nonprob","recommended"},"priority"},
        {{"great_pret","improper","incomplete","more","convenient","convenient","nonprob","priority"},"spec_prior"},
        {{"great_pret","improper","incomplete","more","convenient","convenient","nonprob","not_recom"},"not_recom"},
        {{"great_pret","improper","incomplete","more","convenient","convenient","slightly_prob","recommended"},"priority"},
        {{"great_pret","improper","incomplete","more","convenient","convenient","slightly_prob","priority"},"spec_prior"},
        {{"great_pret","improper","incomplete","more","convenient","convenient","slightly_prob","not_recom"},"not_recom"},
        {{"great_pret","improper","incomplete","more","convenient","convenient","problematic","recommended"},"spec_prior"},
        {{"great_pret","improper","incomplete","more","convenient","convenient","problematic","priority"},"spec_prior"},
        {{"great_pret","improper","incomplete","more","convenient","convenient","problematic","not_recom"},"not_recom"},
        {{"great_pret","improper","incomplete","more","convenient","inconv","nonprob","recommended"},"spec_prior"},
        {{"great_pret","improper","incomplete","more","convenient","inconv","nonprob","priority"},"spec_prior"},
        {{"great_pret","improper","incomplete","more","convenient","inconv","nonprob","not_recom"},"not_recom"},
        {{"great_pret","improper","incomplete","more","convenient","inconv","slightly_prob","recommended"},"spec_prior"},
        {{"great_pret","improper","incomplete","more","convenient","inconv","slightly_prob","priority"},"spec_prior"},
        {{"great_pret","improper","incomplete","more","convenient","inconv","slightly_prob","not_recom"},"not_recom"},
        {{"great_pret","improper","incomplete","more","convenient","inconv","problematic","recommended"},"spec_prior"},
        {{"great_pret","improper","incomplete","more","convenient","inconv","problematic","priority"},"spec_prior"},
        {{"great_pret","improper","incomplete","more","convenient","inconv","problematic","not_recom"},"not_recom"},
        {{"great_pret","improper","incomplete","more","less_conv","convenient","nonprob","recommended"},"spec_prior"},
        {{"great_pret","improper","incomplete","more","less_conv","convenient","nonprob","priority"},"spec_prior"},
        {{"great_pret","improper","incomplete","more","less_conv","convenient","nonprob","not_recom"},"not_recom"},
        {{"great_pret","improper","incomplete","more","less_conv","convenient","slightly_prob","recommended"},"spec_prior"},
        {{"great_pret","improper","incomplete","more","less_conv","convenient","slightly_prob","priority"},"spec_prior"},
        {{"great_pret","improper","incomplete","more","less_conv","convenient","slightly_prob","not_recom"},"not_recom"},
        {{"usual","proper","foster","1","less_conv","inconv","nonprob","recommended"},"priority"},
        {{"usual","proper","foster","1","less_conv","inconv","nonprob","priority"},"priority"},
        {{"usual","proper","foster","1","less_conv","inconv","nonprob","not_recom"},"not_recom"},
        {{"usual","proper","foster","1","less_conv","inconv","slightly_prob","recommended"},"priority"},
        {{"usual","proper","foster","1","less_conv","inconv","slightly_prob","priority"},"priority"},
        {{"usual","proper","foster","1","less_conv","inconv","slightly_prob","not_recom"},"not_recom"},
        {{"usual","proper","foster","1","less_conv","inconv","problematic","recommended"},"priority"},
        {{"usual","proper","foster","1","less_conv","inconv","problematic","priority"},"priority"},
        {{"usual","proper","foster","1","less_conv","inconv","problematic","not_recom"},"not_recom"},
        {{"usual","proper","foster","1","critical","convenient","nonprob","recommended"},"priority"},
        {{"usual","proper","foster","1","critical","convenient","nonprob","priority"},"priority"},
        {{"usual","proper","foster","1","critical","convenient","nonprob","not_recom"},"not_recom"},
        {{"usual","proper","foster","1","critical","convenient","slightly_prob","recommended"},"priority"},
        {{"usual","proper","foster","1","critical","convenient","slightly_prob","priority"},"priority"},
        {{"usual","proper","foster","1","critical","convenient","slightly_prob","not_recom"},"not_recom"},
        {{"usual","proper","foster","1","critical","convenient","problematic","recommended"},"priority"},
       /* {{"usual","proper","foster","2","convenient","convenient","problematic","recommended"},"priority"},
        {{"usual","proper","foster","2","convenient","convenient","problematic","priority"},"priority"},
        {{"usual","proper","foster","2","convenient","convenient","problematic","not_recom"},"not_recom"},
        {{"usual","proper","foster","2","convenient","inconv","nonprob","recommended"},"priority"},
        {{"usual","proper","foster","2","convenient","inconv","nonprob","priority"},"priority"},
        {{"usual","proper","foster","2","convenient","inconv","nonprob","not_recom"},"not_recom"},
        {{"usual","proper","foster","2","convenient","inconv","slightly_prob","recommended"},"priority"},
        {{"usual","proper","foster","2","convenient","inconv","slightly_prob","priority"},"priority"},
        {{"usual","proper","foster","2","convenient","inconv","slightly_prob","not_recom"},"not_recom"},
        {{"usual","proper","foster","2","convenient","inconv","problematic","recommended"},"priority"},
        {{"usual","proper","foster","2","convenient","inconv","problematic","priority"},"priority"},
        {{"usual","proper","foster","2","convenient","inconv","problematic","not_recom"},"not_recom"},
        {{"usual","proper","foster","2","less_conv","convenient","nonprob","recommended"},"priority"},
        {{"usual","proper","foster","2","less_conv","convenient","nonprob","priority"},"priority"},
        {{"usual","proper","foster","2","less_conv","convenient","nonprob","not_recom"},"not_recom"},
        {{"usual","proper","foster","2","less_conv","convenient","slightly_prob","recommended"},"priority"},
        {{"usual","proper","foster","2","less_conv","convenient","slightly_prob","priority"},"priority"},
        {{"usual","proper","foster","2","less_conv","convenient","slightly_prob","not_recom"},"not_recom"},
        {{"usual","proper","foster","2","less_conv","convenient","problematic","recommended"},"priority"},
        {{"usual","proper","foster","2","less_conv","convenient","problematic","priority"},"priority"},
        {{"usual","proper","foster","2","less_conv","convenient","problematic","not_recom"},"not_recom"},
        {{"usual","proper","foster","2","less_conv","inconv","nonprob","recommended"},"priority"},
        {{"usual","proper","foster","2","less_conv","inconv","nonprob","priority"},"priority"},
        {{"usual","proper","foster","2","less_conv","inconv","nonprob","not_recom"},"not_recom"},
        {{"usual","proper","foster","2","less_conv","inconv","slightly_prob","recommended"},"priority"},
        {{"usual","proper","foster","2","less_conv","inconv","slightly_prob","priority"},"priority"},
        {{"usual","proper","foster","2","less_conv","inconv","slightly_prob","not_recom"},"not_recom"},
        {{"usual","proper","foster","2","less_conv","inconv","problematic","recommended"},"priority"},*/
        {{"usual","proper","foster","2","less_conv","inconv","problematic","priority"},"priority"},
        {{"usual","proper","foster","2","less_conv","inconv","problematic","not_recom"},"not_recom"},
        {{"usual","proper","foster","2","critical","convenient","nonprob","recommended"},"priority"},
        {{"usual","proper","foster","2","critical","convenient","nonprob","priority"},"priority"},
        {{"usual","proper","foster","2","critical","convenient","nonprob","not_recom"},"not_recom"},
        {{"usual","proper","foster","2","critical","convenient","slightly_prob","recommended"},"priority"},
        {{"great_pret","improper","foster","1","less_conv","convenient","problematic","priority"},"spec_prior"},
        {{"great_pret","improper","foster","1","less_conv","convenient","problematic","not_recom"},"not_recom"},
        {{"great_pret","improper","foster","1","less_conv","inconv","nonprob","recommended"},"spec_prior"},
        {{"great_pret","improper","foster","1","less_conv","inconv","nonprob","priority"},"spec_prior"},
        {{"great_pret","improper","foster","1","less_conv","inconv","nonprob","not_recom"},"not_recom"},
        {{"great_pret","improper","foster","1","less_conv","inconv","slightly_prob","recommended"},"spec_prior"},
        {{"great_pret","improper","foster","1","less_conv","inconv","slightly_prob","priority"},"spec_prior"},
        {{"great_pret","improper","foster","1","less_conv","inconv","slightly_prob","not_recom"},"not_recom"},
        {{"great_pret","improper","foster","1","less_conv","inconv","problematic","recommended"},"spec_prior"},
        {{"great_pret","improper","foster","1","less_conv","inconv","problematic","priority"},"spec_prior"},
        {{"great_pret","improper","foster","1","less_conv","inconv","problematic","not_recom"},"not_recom"},
        {{"great_pret","improper","foster","1","critical","convenient","nonprob","recommended"},"spec_prior"},
        {{"great_pret","improper","foster","2","critical","inconv","nonprob","recommended"},"spec_prior"},
        {{"great_pret","improper","foster","2","critical","inconv","nonprob","priority"},"spec_prior"},
        {{"great_pret","improper","foster","2","critical","inconv","nonprob","not_recom"},"not_recom"},
        // add more
        };

        vector<string> attributes = { "parents","has_nurs","form","children" ,"housing","finance","social","health" };

        cout << "case 2: " << endl;
        tree* decisionTree = buildDecisionTree(trainingData, attributes, true);   //构造决策树
        printTree(decisionTree);   //打印
        double test1 = predictData(decisionTree, trainingData, false);
        double test2 = predictData(decisionTree, testData, false);
        cout << "训练集的准确度:" << test1 << endl;
        cout << "测试集的准确度:" << test2 << endl;
        vector<string> test = { "great_pret","improper","incomplete","3","critical","inconv","problematic","priority" };
        cout << predictTree(decisionTree, test);

        //输出结果
        freopen("output.txt", "w", stdout);
        cout << "训练集的预测结果" << endl;
        predictData(decisionTree, trainingData, true);
        cout << "测试集的预测结果" << endl;
        predictData(decisionTree, testData, true);

        destroyDecisionTree(decisionTree);
    }

}
