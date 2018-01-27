package processing

import (
	"github.com/DexterLB/search/documents"
	"github.com/DexterLB/search/indices"
)

func CountInDocuments(
	docs <-chan *documents.Document,
	tokeniser Tokeniser,
	idocs chan<- *indices.InfoAndTerms,
	includeClassless bool,
	includeClassy bool,
) {
	for doc := range docs {
		if len(doc.Classes) >= 1 && includeClassy {
			idocs <- Count(doc, tokeniser)
		}
		if len(doc.Classes) == 0 && includeClassless {
			idocs <- Count(doc, tokeniser)
		}
	}
}

func Count(doc *documents.Document, tokeniser Tokeniser) *indices.InfoAndTerms {
	idoc := indices.NewInfoAndTerms()
	idoc.Name = doc.Title
	idoc.Classes = doc.Classes

	tokeniser.GetTerms(doc.Body, func(term string) {
		idoc.TermsAndCounts.PutLambda(
			[]byte(term),
			func(x int32) int32 { return x + 1 },
			1,
		)
		idoc.Length += 1
	})

	return idoc
}
