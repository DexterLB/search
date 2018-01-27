package main

import (
	"log"
	"os"
	"path/filepath"
	"runtime"

	"github.com/DexterLB/search/documents"
	"github.com/DexterLB/search/indices"
	"github.com/DexterLB/search/processing"
	"github.com/DexterLB/search/utils"
	"github.com/urfave/cli"
)

func main() {
	app := cli.NewApp()
	app.Name = "testingtesting"
	app.Usage = "Parse Reuters XML documents and index them"
	app.Flags = []cli.Flag{
		cli.StringFlag{
			Name:  "xmldir, d",
			Usage: "Directory with Reuters XML files",
			Value: ".",
		},
		cli.StringFlag{
			Name:  "stopwords, s",
			Usage: "Stopwords file. If not specified, defaults to ${xmldir}/stopwords",
			Value: "",
		},
		cli.StringFlag{
			Name:  "output, o",
			Usage: "File to write index to",
			Value: "/tmp/index.gob.gz",
		},
		cli.BoolFlag{
			Name:  "classy, y",
			Usage: "Include documents which have >=1 assigned classes",
		},
		cli.BoolFlag{
			Name:  "classless, n",
			Usage: "Include documents which have no assigned class",
		},
		cli.StringFlag{
			Name:  "split",
			Usage: "Split documents into two indices (specify second index as an argument to this option)",
			Value: "",
		},
		cli.StringFlag{
			Name:  "split-size",
			Usage: "Size of first batch of documents (second will be filled with the rest)",
		},
	}

	app.Action = mainCommand

	app.Run(os.Args)
}

func mainCommand(c *cli.Context) {
	files := make(chan string, 200)
	docs := make(chan *documents.Document, 2000)
	infosAndTerms := make(chan *indices.InfoAndTerms, 2000)

	stopWordsFile := c.String("stopwords")
	if stopWordsFile == "" {
		stopWordsFile = filepath.Join(c.String("xmldir"), "stopwords")
	}

	tokeniser, err := processing.NewEnglishTokeniserFromFile(stopWordsFile)
	if err != nil {
		log.Fatal("unable to get stopwords: %s", err)
	}

	go func() {
		GetXMLs(c.String("xmldir"), files)
		close(files)
	}()

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
				c.Bool("classless"),
				c.Bool("classy"),
			)
		}, runtime.NumCPU())
		close(infosAndTerms)
	}()

	index1 := indices.NewTotalIndex()

	if c.String("split") != "" {
		index1.AddUpTo(infosAndTerms, c.Int("split-size"))
	} else {
		index1.AddMany(infosAndTerms)
	}

	if c.String("split") != "" {
		index2 := indices.NewOffsetTotalIndex(index1)
		index2.AddMany(infosAndTerms)
		index2.Verify()

		index1.ExtendInverse(len(index2.Inverse.PostingLists))

		err = index2.SerialiseToFile(c.String("split"))
		if err != nil {
			log.Fatalf("Unable to serialise index: %s", err)
		}
	}

	index1.Verify()

	err = index1.SerialiseToFile(c.String("output"))
	if err != nil {
		log.Fatalf("Unable to serialise index: %s", err)
	}
}

func GetXMLs(folder string, into chan<- string) {
	files, err := filepath.Glob(filepath.Join(folder, "*.xml"))
	if err != nil {
		log.Fatal("unable to get files in folder %s: %s", os.Args[1], err)
	}

	for i := range files {
		into <- files[i]
	}
}
