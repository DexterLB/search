package main

import (
	"log"
	"os"
	"runtime"

	"github.com/DexterLB/search/indices"
	"github.com/DexterLB/search/knn"
	"github.com/DexterLB/search/serialisation"
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
					Name:  "features-per-class, f",
					Usage: "Number of feature terms to select for each class",
					Value: 20,
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

func mainCommand(c *cli.Context) {
}
