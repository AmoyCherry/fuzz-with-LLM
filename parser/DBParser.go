package main

import (
	"github.com/google/syzkaller/pkg/db"
	"github.com/google/syzkaller/pkg/log"
	"os"
	"strconv"
)

type Parser struct {
	corpusDB *db.DB
}

const TokenFilePath = "./tokens/tokens_"
const DBCorpusPath = "./parser/corpus.db"

func NewParser() *Parser {
	corpusDB, err := db.Open(DBCorpusPath, true)
	if err != nil {
		if corpusDB == nil {
			log.Fatalf("failed to open corpus database: %v", err)
		}
		log.Errorf("read %v inputs from corpus and got error: %v", len(corpusDB.Records), err)
	}

	parser := new(Parser)
	parser.corpusDB = corpusDB
	return parser
}

func (parser *Parser) Parse() {
	buffer := ""
	fileCount := 1
	sequenceCount := 0
	totalCount := 0
	for _, rec := range parser.corpusDB.Records {
		buffer += string(rec.Val[:])
		sequenceCount += 1
		totalCount += 1

		if sequenceCount > 10000 {
			parser.WriteToFile(fileCount, &buffer)

			fileCount += 1
			sequenceCount = 0
			buffer = ""
		}
	}
	parser.WriteToFile(fileCount, &buffer)

	log.Logf(0, "total number of sequences (traces): %v", totalCount)
}

func (parser *Parser) WriteToFile(fileCount int, content *string) {
	f, err := os.Create(TokenFilePath + strconv.Itoa(fileCount) + ".txt")
	if err != nil {
		log.Fatalf("open token error :", err)
		return
	}

	_, err = f.Write([]byte(*content))
	if err != nil {
		log.Fatalf("write token error: ", err)
		return
	}

	err = f.Close()
	if err != nil {
		log.Fatalf("Close token error: ", err)
		return
	}
}
