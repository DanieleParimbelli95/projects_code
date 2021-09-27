import streamlit as st

st.title('Classification of Newspaper Web Articles')

author = "*Created by* [*Daniele Parimbelli*](https://danieleparimbelli95.github.io/)"
st.markdown(author)

st.subheader('How Was this App Made?')

st.write("1) Web Scraping")
st.write("2) Summarization & Classification Using a Pretrained BART Model")

st.subheader('How Does it Work?')

st.write("Once the url to a web article is entered, in about 40 seconds the app will return the prediction regarding what the article is about.")

st.write("Possible subjects are: Science, Sports, Politics, Economy/Business, Finance/Stock Market, Crime, War, Technology, Entertainment, Music, Cinema, Television, Theater, Books/Literature, Arts/Architecture, Climate/Environment, Agricolture, Health/Fitness, Covid, Education, Weather, Food/Dining, Style/Fashion, Real Estate, Travel, Cars, Love/Relationships, Video Games, Obituaries, Horoscope, Other.")

st.subheader('Some Things to Keep in Mind')

st.write('a) Classifying into a single subject can be really hard, as the same article might cover more than one of them.')
st.write("b) The model was trained on english samples, so it should only work for articles written in english.")
st.write('c) Not all newspapers can be used (a "HTTP Error 403: Forbidden" error may occur).')

markdown = "d) The app can be used even for web pages that are not from newspapers, but the choice of the topics was developed with that purpose in mind."
st.markdown(markdown)

st.subheader('The App')

url = st.text_input("Enter a valid url")

if url != '':
    
    @st.cache(allow_output_mutation = True, show_spinner = False, suppress_st_warning = True)
    def classify_from_url():
        
        import sys
        from bs4 import BeautifulSoup
        from urllib.request import urlopen
        from transformers import pipeline
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        sys.tracebacklimit = 0

        html = urlopen(url).read()
        soup = BeautifulSoup(html, features = "html.parser")
        
        title_newspaper = soup.find('title').text
        title_list = list(title_newspaper.partition(" - "))
        title_list_sorted = sorted(title_list, key = len)
        title = title_list_sorted[2]
            
        description = soup.find('meta', attrs = {'name': 'description'})
        
        if "content" in str(description):
                description = description.get("content")
        else:
                description = ""
        
        paragraphs_all = ""
        paragraphs = soup.find_all('p')
        
        for x in range (len(paragraphs)):
            if x ==  len(paragraphs) -1:
                paragraphs_all = paragraphs_all + paragraphs[x].text
            else:
                paragraphs_all = paragraphs_all + paragraphs[x].text + ". "
                  
        allthecontent = ""
        allthecontent = str(title) + ". " + str(description) + " " + str(paragraphs_all) 
        allthecontent = str(allthecontent)
        
        clean_content = allthecontent.replace('\t', '').replace('\n', '').replace('Advertisement', '')
        
        
        # SUMMARIZATION
        
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        
        model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
        
        inputs = tokenizer.encode(clean_content, return_tensors =" pt", max_length = 1024, truncation = True)

        outputs = model.generate(inputs, min_length = 50, max_length = 100, length_penalty = 1.0)
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens = True)
        
        
        # CLASSIFICATION
        
        classifier = pipeline("zero-shot-classification", model = "facebook/bart-large-mnli")
        
        candidate_labels = ["Science", "Sports", "Politics", "Economy/Business", "Finance/Stock Market", "Technology",  "Food/Dining", "Entertainment", "Music", "Cinema", "Television", "Theater", "Arts/Architecture", "Books/Literature", "Style/Fashion", "Real Estate", "Travel", "Crime", "Climate/Environment", "Health/Fitness", "Education", "Weather", "Cars", "Love/Relationships", "Video Games", "Obituaries", "Horoscope", "War", "Agricolture", "Covid", "Other"]
        
        hypothesis_template = "The topic is {}."
        
        result = classifier(summary, candidate_labels, hypothesis_template)
        
        pred = result['labels'][0]
        
        return title, pred
     
    st.write('**Title: **', classify_from_url()[0], ' | **Predicted Subject: **', classify_from_url()[1])
