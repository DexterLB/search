package knn

import (
	"fmt"
	"log"
	"strings"

	"github.com/DexterLB/search/indices"
	"github.com/DexterLB/search/trie"
)

func InteractiveTest(classifier func(*DocumentIndex) []int32, testSet *indices.TotalIndex) {
	total := &TestResult{}
	for docID := range testSet.Documents {
		actualClasses := testSet.Documents[docID].Classes
		resultClasses := classifier(&DocumentIndex{
			Postings:    testSet.Forward.Postings,
			PostingList: &testSet.Forward.PostingLists[docID],
		})

		total.Add(Compare(actualClasses, resultClasses, &testSet.ClassNames))
	}

	log.Printf("totals: %s", total)
}

type TestResult struct {
	PrecisionSum     int
	RecallSum        int
	PrecisionDivisor int
	RecallDivisor    int
}

func (t *TestResult) Precision() float64 {
	return float64(t.PrecisionSum) / float64(t.PrecisionDivisor)
}

func (t *TestResult) Recall() float64 {
	return float64(t.RecallSum) / float64(t.RecallDivisor)
}

func (t *TestResult) Add(other *testResult) {
	t.PrecisionSum += other.PresicionSum
	t.RecallSum += other.RecallSum
	t.PrecisionDivisor += other.PrecisionDivisor
	t.RecallDivisor += other.RecallDivisor
}

func (t *TestResult) String() string {
	return fmt.Sprintf("precision: %.2f, recall: %.2f", t.Precision(), t.Recall())
}

func Compare(actualClasses []int32, resultClasses []int32, classDic *trie.BiDictionary) *TestResult {
	tr := &TestResult{}

	actualSet := classSet(actualClasses)
	resultSet := classSet(resultClasses)

	for _, ac := range actualClasses {
		if _, ok := resultSet[ac]; ok {
			tr.RecallSum += 1
		}
		tr.RecallDivisor += 1
	}

	for _, rc := range resultClasses {
		if _, ok := actualSet[rc]; ok {
			tr.PrecisionSum += 1
		} else {
			tr.PrecisionDivisor += 1
		}
	}

	log.Printf("actual: %s", strings.Join(stringifyClasses(actualClasses, classDic), ", "))
	log.Printf("result: %s", strings.Join(stringifyClasses(resultClasses, classDic), ", "))
	log.Printf("%s", tr)

	return tr
}

func classSet(classes []int32) map[int32]struct{} {
	set := make(map[int32]struct{})

	for _, class := range classes {
		set[class] = struct{}{}
	}

	return set
}

func stringifyClasses(classes []int32, dic *trie.BiDictionary) []string {
	s := make([]string, len(classes))

	for i := range classes {
		s[i] = string(dic.GetInverse(classes[i]))
	}

	return s
}
