package indices

import (
	"fmt"
	"log"
	"sort"

	"github.com/DexterLB/search/trie"
)

type InfoAndTerms struct {
	Name           string
	Classes        []string
	Length         int32
	TermsAndCounts trie.Trie
}

func NewInfoAndTerms() *InfoAndTerms {
	return &InfoAndTerms{
		TermsAndCounts: *trie.New(),
	}
}

type TermAndCount struct {
	TermID int32
	Count  int32
}

func (d *InfoAndTerms) Print() {
	fmt.Printf("************\n")
	fmt.Printf(
		"name: %s, classes: %v, length: %d\nterms:\n",
		d.Name,
		d.Classes,
		d.Length,
	)
	d.TermsAndCounts.Walk(func(term []byte, count int32) {
		fmt.Printf("  %s: %d\n", string(term), count)
	})
}

func (t *TotalIndex) AddMany(infosAndTerms <-chan *InfoAndTerms) {
	t.AddUpTo(infosAndTerms, -1)
}

func (t *TotalIndex) AddUpTo(infosAndTerms <-chan *InfoAndTerms, limit int) {
	for it := range infosAndTerms {
		if limit > 0 {
			limit--
		}
		if limit == 0 {
			return
		}

		if it.TermsAndCounts.Empty() {
			log.Printf("Document %s is empty", it.Name)
		} else {
			t.Add(it)
		}
	}
}

func (t *TotalIndex) Add(d *InfoAndTerms) {
	var sortedTermsAndCounts []TermAndCount

	d.TermsAndCounts.Walk(func(term []byte, count int32) {
		sortedTermsAndCounts = append(
			sortedTermsAndCounts,
			TermAndCount{
				TermID: t.Dictionary.Get(term),
				Count:  count,
			},
		)
	})

	sort.Slice(
		sortedTermsAndCounts,
		func(i, j int) bool {
			return sortedTermsAndCounts[i].TermID < sortedTermsAndCounts[j].TermID
		},
	)

	//d
	//sortedTermsAndCount

	documentIndex := int32(len(t.Documents))
	info := DocumentInfo{
		Name:   d.Name,
		Length: d.Length,
	}

	info.Classes = make([]int32, len(d.Classes))
	for i := range d.Classes {
		info.Classes[i] = t.ClassNames.Get([]byte(d.Classes[i]))
	}

	t.Documents = append(t.Documents, info)

	// d0
	// <- d1
	// t.Postinglist = [f:0 l:1] ->
	// t.Postings = [["foo" 0 2 1], ["bar" 1 2 -1]]
	//
	// t.Postinglist = [f:0 l:1] -> [f:2, l:2]
	// <- bar
	// t.Postings = [["foo" 0 2 1], ["bar" 1 2 -1] -> ["bar" 1 1 -1]
	// <- qux
	// t.Postinglist = [f:0 l:1] -> [f:2, l:3]
	// t.Postings = [["foo" 0 2 1], ["bar" 1 2 -1] -> ["bar" 1 1 3] -> ["qux" 2 1 -1]
	//
	//

	t.Forward.PostingLists = append(t.Forward.PostingLists, PostingList{FirstIndex: -1, LastIndex: -1})

	for _, term := range sortedTermsAndCounts {
		// Forward indexing
		t.Forward.Postings = append(t.Forward.Postings, Posting{Index: term.TermID, Count: term.Count, NextPostingIndex: -1})

		if t.Forward.PostingLists[documentIndex].FirstIndex == -1 {
			t.Forward.PostingLists[documentIndex].FirstIndex = int32(len(t.Forward.Postings) - 1)
			t.Forward.PostingLists[documentIndex].LastIndex = int32(len(t.Forward.Postings) - 1)
		} else {

			t.Forward.Postings[t.Forward.PostingLists[documentIndex].LastIndex].NextPostingIndex = int32(len(t.Forward.Postings) - 1)
			t.Forward.PostingLists[documentIndex].LastIndex += 1
		}

		//Inverse indexing
		if int32(len(t.Inverse.PostingLists)) > term.TermID {
			// already in

			t.Inverse.Postings = append(t.Inverse.Postings, Posting{Index: documentIndex, Count: term.Count, NextPostingIndex: -1})
			if t.Inverse.PostingLists[term.TermID].FirstIndex == -1 {
				// posting list is empty
				t.Inverse.PostingLists[term.TermID].FirstIndex = int32(len(t.Inverse.Postings)) - 1
			} else {
				t.Inverse.Postings[t.Inverse.PostingLists[term.TermID].LastIndex].NextPostingIndex = int32(len(t.Inverse.Postings)) - 1
			}
			t.Inverse.PostingLists[term.TermID].LastIndex = int32(len(t.Inverse.Postings)) - 1
		} else if int32(len(t.Inverse.PostingLists)) == term.TermID {
			//PL -> [f:0 l:0]
			// [index: 0, count: 2, NextPI: -1]
			t.Inverse.PostingLists = append(t.Inverse.PostingLists, PostingList{FirstIndex: int32(len(t.Inverse.Postings)), LastIndex: int32(len(t.Inverse.Postings))})
			t.Inverse.Postings = append(t.Inverse.Postings, Posting{Index: documentIndex, Count: term.Count, NextPostingIndex: -1})
		} else {
			panic(fmt.Sprintf("lenPostingLists: %d, lenPostings: %d, termId: %d\n", len(t.Inverse.PostingLists), len(t.Inverse.Postings), term.TermID))
		}

	}

}
