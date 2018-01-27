package knn

import (
	"math"
	"sort"

	"github.com/DexterLB/search/featureselection"
	"github.com/DexterLB/search/indices"
	"github.com/DexterLB/search/utils"
)

type DocumentIndex struct {
	Postings    []indices.Posting
	PostingList *indices.PostingList
}

type KNNInfo struct {
	Features []int32
	Index    *indices.TotalIndex
}

func Preprocess(ti *indices.TotalIndex, termsPerClass int32, parallelWorkers int) *KNNInfo {
	return &KNNInfo{
		Features: featureselection.ChiSquared(ti, termsPerClass, parallelWorkers),
		Index:    ti,
	}
}

type DocumentDistance struct {
	DocumentID int32
	Distance   float64
}

func (k *KNNInfo) ClassifyForward(document *DocumentIndex, bestK int, parallelWorkers int) []int32 {
	distances := make(chan *DocumentDistance, 200)
	go func() {
		k.forwardDistances(document, parallelWorkers, distances)
		close(distances)
	}()
	return k.bestClasses(distances, bestK)
}

func (k *KNNInfo) bestClasses(distances <-chan *DocumentDistance, bestK int) []int32 {
	var allDistances []*DocumentDistance
	for dist := range distances {
		allDistances = append(allDistances, dist)
	}

	// todo: use priority queue instead of this

	sort.Slice(allDistances, func(i, j int) bool {
		return allDistances[i].Distance < allDistances[j].Distance
	})

	bestDistances := allDistances[0:bestK]

	classHistogram := make(map[int32]int)
	for i := range bestDistances {
		for _, class := range k.Index.Documents[bestDistances[i].DocumentID].Classes {
			if _, ok := classHistogram[class]; ok {
				classHistogram[class] += 1
			} else {
				classHistogram[class] = 1
			}
		}
	}

	mostOccurrences := -1
	for _, occurrences := range classHistogram {
		if occurrences > mostOccurrences {
			mostOccurrences = occurrences
		}
	}

	var bestClasses []int32
	for class, occurrences := range classHistogram {
		if occurrences == mostOccurrences {
			bestClasses = append(bestClasses, class)
		}
	}

	sort.Slice(bestClasses, func(i, j int) bool { return bestClasses[i] < bestClasses[j] })

	return bestClasses
}

func (k *KNNInfo) forwardDistances(document *DocumentIndex, parallelWorkers int, distances chan<- *DocumentDistance) {
	docsToProcess := make(chan int32, 200)

	go func() {
		for docID := range k.Index.Forward.PostingLists {
			docsToProcess <- int32(docID)
		}

		close(docsToProcess)
	}()

	utils.Parallel(
		func() {
			for docID := range docsToProcess {
				distances <- &DocumentDistance{
					Distance: k.distance(document, &DocumentIndex{
						Postings:    k.Index.Forward.Postings,
						PostingList: &k.Index.Forward.PostingLists[docID],
					}),
					DocumentID: docID,
				}
			}
		},
		parallelWorkers,
	)
}

func (k *KNNInfo) distance(a *DocumentIndex, b *DocumentIndex) float64 {
	postingAindex := a.PostingList.FirstIndex
	postingBindex := b.PostingList.FirstIndex

	dist := float64(0)

	for _, featureID := range k.Features {
		for postingAindex >= 0 && a.Postings[postingAindex].Index < featureID {
			postingAindex = a.Postings[postingAindex].NextPostingIndex
		}
		for postingBindex >= 0 && b.Postings[postingBindex].Index < featureID {
			postingBindex = b.Postings[postingBindex].NextPostingIndex
		}

		if postingAindex == -1 && postingBindex == -1 {
			break
		}

		termA := float64(0)
		termB := float64(0)

		if postingAindex >= 0 {
			postingA := &a.Postings[postingAindex]
			if postingA.Index == featureID {
				termA = float64(postingA.Count)
			}
		}

		if postingBindex >= 0 {
			postingB := &b.Postings[postingBindex]
			if postingB.Index == featureID {
				termB = float64(postingB.Count)
			}
		}

		dist += square(termA - termB)
	}

	return dist
}

func (k *KNNInfo) distanceToAll(document *DocumentIndex, distances chan<- DocumentDistance) {
	postingLists := k.Index.Inverse.PostingLists
	postings := k.Index.Inverse.Postings

	docVec := k.documentVector(document)

	numFeatures := len(k.Features)

	currentPostingIndices := make([]int32, numFeatures)
	for i := 0; i < numFeatures; i += 1 {
		currentPostingIndices[i] = postingLists[k.Features[i]].FirstIndex
	}

	docIndex := int32(0)
	for docIndex != int32(math.MaxInt32) {
		minDocIndex := int32(math.MaxInt32)
		distance := float64(0)

		for i := 0; i < numFeatures; i += 1 {
			posting := &postings[currentPostingIndices[i]]
			if posting.Index == docIndex {
				distance += square(float64(posting.Count) - docVec[i])

				currentPostingIndices[i] = posting.NextPostingIndex
			}
			if postings[currentPostingIndices[i]].Index < minDocIndex {
				minDocIndex = postings[currentPostingIndices[i]].Index
			}
		}

		distances <- DocumentDistance{
			DocumentID: docIndex,
			Distance:   distance,
		}

		docIndex = minDocIndex
	}
}

func (k *KNNInfo) documentVector(document *DocumentIndex) []float64 {
	docVec := make([]float64, len(k.Features))

	postingIndex := document.PostingList.FirstIndex
	for i, featureIndex := range k.Features {
		for postingIndex >= 0 && document.Postings[postingIndex].Index < featureIndex {
			postingIndex = document.Postings[postingIndex].NextPostingIndex
		}

		if document.Postings[postingIndex].Index == featureIndex {
			docVec[i] = float64(document.Postings[postingIndex].Count)
		}
	}

	return docVec
}

func square(x float64) float64 {
	return x * x
}
