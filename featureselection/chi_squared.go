package featureselection

import (
	"sort"

	"github.com/DexterLB/search/indices"
	"github.com/DexterLB/search/utils"
)

type TermScore struct {
	TermID int32
	Score  float64
}

func SortedChiSquaredTable(ti *indices.TotalIndex, ci *ClassInfo, parallelWorkers int) [][]TermScore {
	table := make([][]TermScore, ci.NumClasses)

	work := make(chan int32, 2000)
	go func() {
		for i := int32(0); i < ci.NumClasses; i++ {
			work <- i
		}
		close(work)
	}()

	numTerms := int32(len(ti.Inverse.PostingLists))

	utils.Parallel(func() {
		for classID := range work {
			table[classID] = make([]TermScore, numTerms)
			for termID := int32(0); termID < numTerms; termID++ {
				table[classID][termID].TermID = termID
				table[classID][termID].Score = ChiSquaredForTermAndClass(ti, ci, termID, classID)
			}
			sortScores(table[classID])
		}
	}, parallelWorkers)

	return table
}

func sortScores(scores []TermScore) {
	sort.Slice(scores, func(i, j int) bool { return scores[i].Score < scores[j].Score })
}

func ChiSquaredForClass(ti *indices.TotalIndex, ci *ClassInfo, classID int32) []float64 {
	termChiSquared := make([]float64, len(ti.Inverse.PostingLists))
	for i := range termChiSquared {
		termID := int32(i)
		termChiSquared[termID] = ChiSquaredForTermAndClass(ti, ci, termID, classID)
	}
	return termChiSquared
}

func ChiSquaredForTermAndClass(ti *indices.TotalIndex, ci *ClassInfo, termID int32, classID int32) float64 {
	numDocuments := int32(len(ti.Forward.PostingLists))

	var N00 int32 // Documents which DON'T contain the term and DON'T have the class
	var N01 int32 // Documents which DO    contain the term and DON'T have the class
	var N10 int32 // Documents which DON'T contain the term and DO    have the class
	var N11 int32 // Documents which DO    contain the term and DO    have the class

	ti.LoopOverTermPostings(int(termID), func(posting *indices.Posting) {
		if docHasClass(ti, posting.Index, classID) {
			N11 += 1
		} else {
			N01 += 1
		}
	})

	N10 = ci.DocumentsWhichContainTerm[termID] - N11
	N00 = numDocuments - N01 - N10 + N11

	N := float64(numDocuments)

	E11 := (float64(N11+N10) * float64(N11+N01)) / N
	E01 := (float64(N01+N00) * float64(N11+N01)) / N
	E10 := (float64(N11+N10) * float64(N10+N00)) / N
	E00 := (float64(N01+N00) * float64(N10+N00)) / N

	M00 := square(float64(N00)-E00) / N
	M01 := square(float64(N01)-E01) / N
	M10 := square(float64(N10)-E10) / N
	M11 := square(float64(N11)-E11) / N

	return M00 + M01 + M10 + M11
}

func docHasClass(ti *indices.TotalIndex, docID int32, classID int32) bool {
	for _, class := range ti.Documents[docID].Classes {
		if class == classID {
			return true
		}
	}
	return false
}

func square(x float64) float64 {
	return x * x
}