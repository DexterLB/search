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
	}

	app.Run(os.Args)
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
