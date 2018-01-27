package indices

import (
	"fmt"
	"io"

	"github.com/DexterLB/search/serialisation"
	"github.com/DexterLB/search/trie"
)

type Posting struct {
	Index int32
	Count int32

	NextPostingIndex int32
}

type PostingList struct {
	FirstIndex int32
	LastIndex  int32
}

type Index struct {
	PostingLists []PostingList
	Postings     []Posting
}

type TotalIndex struct {
	Forward    Index
	Inverse    Index
	Documents  []DocumentInfo
	Dictionary *trie.BiDictionary // bidictionary is better for debugging
	ClassNames *trie.BiDictionary
}

type DocumentInfo struct {
	Name    string
	Classes []int32
	Length  int32
}

func NewTotalIndex() *TotalIndex {
	return &TotalIndex{
		Dictionary: trie.NewBiDictionary(),
		ClassNames: trie.NewBiDictionary(),
	}
}

// NewOffsetTotalIndex creates a TotalIndex with the same dictionaries as the first one
func NewOffsetTotalIndex(other *TotalIndex) *TotalIndex {
	ni := NewTotalIndex()
	ni.Dictionary = other.Dictionary
	ni.ClassNames = other.ClassNames
	ni.ExtendInverse(len(other.Inverse.PostingLists))

	return ni
}

func (t *TotalIndex) ExtendInverse(newLength int) {
	for i := len(t.Inverse.PostingLists); i < newLength; i++ {
		t.Inverse.PostingLists = append(t.Inverse.PostingLists, PostingList{
			FirstIndex: -1,
			LastIndex:  -1,
		})
	}
}

func (t *TotalIndex) LoopOverTermPostings(termID int, operation func(posting *Posting)) {
	postingList := &t.Inverse.PostingLists[termID]
	if postingList.FirstIndex == -1 {
		return
	}

	for posting := &t.Inverse.Postings[postingList.FirstIndex]; ; posting = &t.Inverse.Postings[posting.NextPostingIndex] {
		operation(posting)

		if posting.NextPostingIndex == -1 {
			break
		}
	}
}

func (t *TotalIndex) LoopOverDocumentPostings(docID int, operation func(posting *Posting)) {
	if docID == -1 {
		panic("DocID index is -1\n")
	}

	if docID >= len(t.Forward.PostingLists) {
		panic(fmt.Sprintf("DocID: %d, size of PostingLists: %d", docID, len(t.Forward.PostingLists)))
	}

	postingList := &t.Forward.PostingLists[docID]
	if postingList.FirstIndex == -1 {
		return
	}

	if postingList.FirstIndex == -1 {
		fmt.Printf("DocId has first index -1: %d, Postinglist: %v\n", docID, postingList)
		return
	}

	for posting := &t.Forward.Postings[postingList.FirstIndex]; ; posting = &t.Forward.Postings[posting.NextPostingIndex] {
		operation(posting)
		if posting.NextPostingIndex == -1 {
			break
		}
	}
}

func (t *TotalIndex) SerialiseTo(w io.Writer) error {
	return serialisation.SerialiseTo(t, w)
}

func (t *TotalIndex) SerialiseToFile(filename string) error {
	return serialisation.SerialiseToFile(t, filename)
}

func (t *TotalIndex) DeserialiseFrom(r io.Reader) error {
	return serialisation.DeserialiseFrom(t, r)
}

func (t *TotalIndex) DeserialiseFromFile(filename string) error {
	return serialisation.DeserialiseFromFile(t, filename)
}

func (t *TotalIndex) Verify() {
	for docID := range t.Forward.PostingLists {
		var lastPosting *Posting
		t.LoopOverDocumentPostings(docID, func(posting *Posting) {
			if lastPosting != nil {
				if posting.Index <= lastPosting.Index {
					panic(fmt.Sprintf(
						"consecutive postings of document %d have out of order term indices: %d, %d",
						docID, lastPosting.Index, posting.Index,
					))
				}
			}
			lastPosting = posting
		})
		if lastPosting == nil {
			panic(fmt.Sprintf("document %d has no terms", docID))
		}
	}

	for termID := range t.Inverse.PostingLists {
		var lastPosting *Posting
		t.LoopOverTermPostings(termID, func(posting *Posting) {
			if lastPosting != nil {
				if posting.Index <= lastPosting.Index {
					panic(fmt.Sprintf(
						"consecutive postings of term %d have out of order document indices: %d, %d",
						termID, lastPosting.Index, posting.Index,
					))
				}
			}
			lastPosting = posting
		})
	}
}
