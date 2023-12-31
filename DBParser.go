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

func NewParser() *Parser {
	corpusDB, err := db.Open("./corpus.db", true)
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
	tokenCount := 0
	for _, rec := range parser.corpusDB.Records {
		buffer += string(rec.Val[:])
		tokenCount += 1

		if tokenCount > 10000 {
			parser.WriteToFile(fileCount, &buffer)

			fileCount += 1
			tokenCount = 0
			buffer = ""
		}

	}
	parser.WriteToFile(fileCount, &buffer)
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
