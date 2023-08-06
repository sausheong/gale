package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"text/template"

	"github.com/go-chi/chi"
	"github.com/go-chi/chi/middleware"
	"github.com/joho/godotenv"
	"github.com/tmc/langchaingo/chains"
	openaiEmbeddings "github.com/tmc/langchaingo/embeddings/openai"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/pinecone"
)

var store pinecone.Store

// initialise to load environment variable from .env file
// and to create a vector store
func init() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}
	embeddings, err := openaiEmbeddings.NewOpenAI()
	if err != nil {
		log.Fatal("Error getting embeddings", err)
	}
	store, err = pinecone.New(
		context.Background(),
		pinecone.WithAPIKey(os.Getenv("PINECONE_API_KEY")),
		pinecone.WithEnvironment(os.Getenv("PINECONE_ENV")),
		pinecone.WithIndexName(os.Getenv("PINECODE_INDEX")),
		pinecone.WithProjectName(os.Getenv("PINECODE_PROJECT")),
		pinecone.WithEmbedder(embeddings),
	)
	if err != nil {
		log.Fatal("Error loading setting up Pinecone store", err)
	}
}

func main() {
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Handle("/static/*", http.StripPrefix("/static", http.FileServer(http.Dir("./static"))))
	r.Get("/", index)
	r.Post("/run", run)
	log.Println("\033[93mGale started. Press CTRL+C to quit.\033[0m")
	http.ListenAndServe(":"+os.Getenv("PORT"), r)
}

func index(w http.ResponseWriter, r *http.Request) {
	t, _ := template.ParseFiles("static/index.html")
	t.Execute(w, nil)
}

func run(w http.ResponseWriter, r *http.Request) {
	prompt := struct {
		Input string `json:"input"`
	}{}

	err := json.NewDecoder(r.Body).Decode(&prompt)
	if err != nil {
		log.Println(err.Error())
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	llm, err := openai.NewChat(openai.WithModel(os.Getenv("OPENAI_MODEL")))
	if err != nil {
		log.Println(err.Error())
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	retriever := vectorstores.ToRetriever(store, 3)
	mem := memory.NewConversationTokenBuffer(llm, 2048)
	chain := chains.NewConversationalRetrievalQAFromLLM(llm, retriever, mem)
	result, err := chains.Run(
		context.TODO(),
		chain,
		prompt.Input,
		chains.WithMaxTokens(2048),
		chains.WithMinLength(512),
	)
	if err != nil {
		log.Println(err.Error())
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	response := struct {
		Input    string `json:"input"`
		Response string `json:"response"`
	}{
		Input:    prompt.Input,
		Response: result,
	}
	json.NewEncoder(w).Encode(response)
}
