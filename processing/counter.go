package processing

import (
	"github.com/DexterLB/search/documents"
	"github.com/DexterLB/search/indices"
)

func CountInDocuments(docs <-chan *documents.Document, tokeniser Tokeniser, idocs chan<- *indices.Document) {
	for doc := range docs {
		idocs <- Count(doc, tokeniser)
	}
}

func Count(doc *documents.Document, tokeniser Tokeniser) *indices.Document {
	idoc := indices.NewDocument()
	idoc.Name = doc.Title
	idoc.Classes = doc.Classes

	tokeniser.GetTerms(doc.Body, func(term string) {
		idoc.TermCounts.PutLambda(
			[]byte(term),
			func(x uint64) uint64 { return x + 1 },
			1,
		)
	})

	return idoc
}