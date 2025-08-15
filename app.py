from flask import Flask, request, render_template
import google.generativeai as genai
import re
import os

# Configure Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDgSJYC8jrqEBhFVjAdC4TclBtGhP1ulnc"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

app = Flask(__name__, template_folder='front')

gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_diet_recommendation(age, gender, weight, height, veg_or_nonveg, disease, region, allergics, foodtype):
    prompt = f"""Diet Recommendation System:
Please provide exactly 6 restaurant names, 6 breakfast items, 5 dinner items, and 6 workout exercises based on the following criteria:

Person Details:
- Age: {age}
- Gender: {gender}
- Weight: {weight} kg
- Height: {height} m
- Diet Type: {veg_or_nonveg}
- Health Condition: {disease}
- Region: {region}
- Allergies: {allergics}
- Preferred Food Type: {foodtype}

Please format your response exactly as follows:

**Restaurants:**
1. Restaurant name 1
2. Restaurant name 2
3. Restaurant name 3
4. Restaurant name 4
5. Restaurant name 5
6. Restaurant name 6

**Breakfast:**
1. Breakfast item 1
2. Breakfast item 2
3. Breakfast item 3
4. Breakfast item 4
5. Breakfast item 5
6. Breakfast item 6

**Dinner:**
1. Dinner item 1
2. Dinner item 2
3. Dinner item 3
4. Dinner item 4
5. Dinner item 5

**Workouts:**
1. Workout exercise 1
2. Workout exercise 2
3. Workout exercise 3
4. Workout exercise 4
5. Workout exercise 5
6. Workout exercise 6"""
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'GET':
        # If someone tries to access /recommend directly, redirect to home
        return render_template('index.html')
    
    try:
        age = request.form.get('age')
        gender = request.form.get('gender')
        weight = request.form.get('weight')
        height = request.form.get('height')
        veg_or_nonveg = request.form.get('veg_or_nonveg')
        disease = request.form.get('disease', 'None')
        region = request.form.get('region')
        allergics = request.form.get('allergics', 'None')
        foodtype = request.form.get('foodtype')

        # Generate recommendations using Gemini
        gemini_result = get_gemini_diet_recommendation(age, gender, weight, height, veg_or_nonveg, disease, region, allergics, foodtype)
        
        print("Gemini Result:", gemini_result)  # Debug print
        
        # Improved parsing for the formatted response
        restaurant_names = []
        breakfast_names = []
        dinner_names = []
        workout_names = []
        
        # Split the response into sections
        sections = gemini_result.split('**')
        current_section = ""
        
        for section in sections:
            if 'Restaurants:' in section or 'Restaurant' in section:
                current_section = "restaurants"
                continue
            elif 'Breakfast:' in section or 'Breakfast' in section:
                current_section = "breakfast"
                continue
            elif 'Dinner:' in section or 'Dinner' in section:
                current_section = "dinner"
                continue
            elif 'Workouts:' in section or 'Workout' in section:
                current_section = "workouts"
                continue
            
            # Extract items from current section
            if current_section and section.strip():
                lines = section.strip().split('\n')
                items = []
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.')) or 
                               line.startswith(('-', '•')) or 
                               (not line.startswith('**') and len(line) > 2)):
                        # Clean the line
                        item = re.sub(r'^\d+\.\s*', '', line)  # Remove numbers
                        item = re.sub(r'^[-•]\s*', '', item)   # Remove bullets
                        item = item.strip()
                        if item:
                            items.append(item)
                
                if current_section == "restaurants":
                    restaurant_names.extend(items)
                elif current_section == "breakfast":
                    breakfast_names.extend(items)
                elif current_section == "dinner":
                    dinner_names.extend(items)
                elif current_section == "workouts":
                    workout_names.extend(items)
        
        # Fallback parsing if the above doesn't work
        if not any([restaurant_names, breakfast_names, dinner_names, workout_names]):
            lines = gemini_result.split('\n')
            current_list = None
            
            for line in lines:
                line = line.strip()
                if 'restaurant' in line.lower():
                    current_list = restaurant_names
                elif 'breakfast' in line.lower():
                    current_list = breakfast_names
                elif 'dinner' in line.lower():
                    current_list = dinner_names
                elif 'workout' in line.lower():
                    current_list = workout_names
                elif line and current_list is not None and not line.startswith('**'):
                    # Clean and add the item
                    item = re.sub(r'^\d+\.\s*', '', line)
                    item = re.sub(r'^[-•]\s*', '', item)
                    item = item.strip()
                    if item and len(current_list) < 6:  # Limit items
                        current_list.append(item)
        
        # Ensure we have some data (fallback)
        if not restaurant_names:
            restaurant_names = ["Local Health Restaurant", "Fresh Food Cafe", "Nutrition Hub", "Healthy Bites", "Green Garden Restaurant", "Fit Food Corner"]
        if not breakfast_names:
            breakfast_names = ["Oatmeal with fruits", "Greek yogurt with nuts", "Whole grain toast", "Smoothie bowl", "Egg white omelet", "Fresh fruit salad"]
        if not dinner_names:
            dinner_names = ["Grilled chicken salad", "Vegetable stir-fry", "Quinoa bowl", "Fish with vegetables", "Lentil soup"]
        if not workout_names:
            workout_names = ["Morning walk", "Light jogging", "Basic stretching", "Yoga session", "Bodyweight exercises", "Swimming"]
        
        print("Parsed results:")  # Debug prints
        print("Restaurants:", restaurant_names)
        print("Breakfast:", breakfast_names)
        print("Dinner:", dinner_names)
        print("Workouts:", workout_names)
        
        return render_template('result.html', 
                               restaurant_names=restaurant_names, 
                               breakfast_names=breakfast_names, 
                               dinner_names=dinner_names, 
                               workout_names=workout_names)
    
    except Exception as e:
        print(f"Error in recommend route: {str(e)}")
        # Return error page or redirect with error
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
