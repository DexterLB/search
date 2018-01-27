package main

import (
	"log"
	"os"
	"runtime"

	"github.com/DexterLB/search/documents"
	"github.com/DexterLB/search/indices"
	"github.com/DexterLB/search/knn"
	"github.com/DexterLB/search/processing"
	"github.com/DexterLB/search/serialisation"
	"github.com/DexterLB/search/utils"
	"github.com/urfave/cli"
)

func main() {
	app := cli.NewApp()
	app.Name = "knn"
	app.Usage = "Perform kNN"

	app.Commands = []cli.Command{
		{
			Name:   "preprocess",
			Usage:  "preprocess an index to create a KNN Info file",
			Action: preprocess,
			Flags: []cli.Flag{
				cli.StringFlag{
					Name:  "input, i",
					Usage: "File with index",
					Value: "/tmp/index.gob.gz",
				},
				cli.StringFlag{
					Name:  "output, o",
					Usage: "Preprocessed data",
					Value: "/tmp/knn.gob.gz",
				},
				cli.IntFlag{
					Name:  "features-per-class, f",
					Usage: "Number of feature terms to select for each class",
					Value: 20,
				},
			},
		},
		{
			Name:   "classify-reuters",
			Usage:  "classify a single file with documents",
			Action: classifyReuters,
			Flags: []cli.Flag{
				cli.StringFlag{
					Name:  "data, d",
					Usage: "Preprocessed data",
					Value: "/tmp/knn.gob.gz",
				},
				cli.StringFlag{
					Name:  "input, i",
					Usage: "Input XML file",
					Value: "/tmp/foo.xml",
				},
				cli.IntFlag{
					Name:  "features-per-class, f",
					Usage: "Number of feature terms to select for each class",
					Value: 20,
				},
				cli.IntFlag{
					Name:  "k",
					Usage: "Number of neighbours to consider for classification",
					Value: 3,
				},
				cli.StringFlag{
					Name:  "stopwords, s",
					Usage: "Stopwords file",
					Value: "",
				},
			},
		},
		{
			Name:   "test",
			Usage:  "perform a test with a split index",
			Action: test,
			Flags: []cli.Flag{
				cli.StringFlag{
					Name:  "training-set",
					Usage: "Training set index",
					Value: "/tmp/index.gob.gz",
				},
				cli.StringFlag{
					Name:  "test-set",
					Usage: "Test set index",
					Value: "/tmp/index_test.gob.gz",
				},
				cli.IntFlag{
					Name:  "k",
					Usage: "Number of neighbours to consider for classification",
					Value: 3,
				},
			},
		},
	}

	app.Run(os.Args)
}

func test(c *cli.Context) {
	numCPU := runtime.NumCPU()
	trainingSet := indices.NewTotalIndex()
	err := trainingSet.DeserialiseFromFile(c.String("training-set"))
	if err != nil {
		log.Fatal(err)
	}

	testSet := indices.NewTotalIndex()
	err = testSet.DeserialiseFromFile(c.String("test-set"))
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("begin preprocessing")
	ki := knn.Preprocess(trainingSet, int32(c.Int("features-per-class")), numCPU)
	log.Printf("end preprocessing")

	k := c.Int("k")

	classifier := func(document *knn.DocumentIndex) []int32 {
		forward := ki.ClassifyForward(document, k, numCPU)
		// inverse := ki.ClassifyInverse(document, k)

		// if len(forward) != len(inverse) {
		// 	panic("different number of classes for forward and inverse classifier")
		// }
		// for i := range forward {
		// 	if forward[i] != inverse[i] {
		// 		panic("different classes for forward and inverse classifier")
		// 	}
		// }

		// return inverse
		return forward
	}

	knn.InteractiveTest(classifier, testSet)
}

func classifyReuters(c *cli.Context) {
	ki := &knn.KNNInfo{}
	err := serialisation.DeserialiseFromFile(ki, c.String("data"))
	if err != nil {
		log.Fatal(err)
	}

	tokeniser, err := processing.NewEnglishTokeniserFromFile(c.String("stopwords"))
	if err != nil {
		log.Fatal("unable to get stopwords: %s", err)
	}

	files := make(chan string, 1)
	files <- c.String("input")
	close(files)
	docs := make(chan *documents.Document, 2000)
	infosAndTerms := make(chan *indices.InfoAndTerms, 2000)

	go func() {
		utils.Parallel(func() {
			documents.NewReutersParser().ParseFiles(files, docs)
		}, runtime.NumCPU())
		close(docs)
	}()

	go func() {
		utils.Parallel(func() {
			processing.CountInDocuments(
				docs,
				tokeniser,
				infosAndTerms,
				true,
				true,
			)
		}, runtime.NumCPU())
		close(infosAndTerms)
	}()

	newDocsIndex := indices.NewOffsetTotalIndex(ki.Index)
	newDocsIndex.AddMany(infosAndTerms)

	interactiveClassify(ki, newDocsIndex, c.Int("k"))
}

func preprocess(c *cli.Context) {
	ti := indices.NewTotalIndex()
	err := ti.DeserialiseFromFile(c.String("input"))
	if err != nil {
		log.Fatal(err)
	}

	ki := knn.Preprocess(ti, int32(c.Int("features-per-class")), runtime.NumCPU())

	err = serialisation.SerialiseToFile(ki, c.String("output"))
	if err != nil {
		log.Fatal(err)
	}
}

func interactiveClassify(ki *knn.KNNInfo, ti *indices.TotalIndex, k int) {
	for docID := range ti.Forward.PostingLists {
		docIndex := &knn.DocumentIndex{
			PostingList: &ti.Forward.PostingLists[docID],
			Postings:    ti.Forward.Postings,
			Length:      ti.Documents[docID].Length,
		}

		classes := ki.ClassifyForward(docIndex, k, runtime.NumCPU())
		log.Printf("document %s\n  --> %s", ti.Documents[docID].Name, ti.StringifyClasses(classes))
	}
}
