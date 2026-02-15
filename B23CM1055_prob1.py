import re
import sys
from datetime import date

def calculate_age(birth_date_str):
    """
    Calculates age using regular expressions to parse various date formats.
    Supported: dd-mm-yyyy, dd-mm-yy, yyyy-mm-dd, dd Month yyyy, Month dd yyyy, and space-separated numeric.
    """
    today = date.today()
    # Updated patterns to handle hyphens and spaces with 2 or 4 digit years
    patterns = [
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', 'ymd'),      # 1984-01-16
        (r'(\d{1,2})[-\s](\d{1,2})[-\s](\d{2,4})', 'numeric'), # 16-01-84, 31 12 2010, 12 31 64
        (r'(\d{1,2})\s+([A-Za-z]+)\s+(\d{2,4})', 'dMy'), # 16 Jan 84 or 16 Jan 1984
        (r'([A-Za-z]+)\s+(\d{1,2})\s+(\d{2,4})', 'Mdy')  # January 16 1984
    ]
    
    for pattern, p_type in patterns:
        match = re.search(pattern, birth_date_str.strip(), re.IGNORECASE)
        if match:
            y, m, d = 0, 0, 0
            
            if p_type == 'ymd':
                y_str, m_str, d_str = match.groups()
                y, m, d = int(y_str), int(m_str), int(d_str)
            elif p_type == 'numeric':
                val1, val2, y_str = match.groups()
                v1, v2, y = int(val1), int(val2), int(y_str)
                # [cite_start]Logic to distinguish between DD MM and MM DD 
                if v1 > 12:  # Must be Day-Month (e.g., 31 12)
                    d, m = v1, v2
                elif v2 > 12: # Must be Month-Day (e.g., 12 31)
                    m, d = v1, v2
                else: # Ambiguous (e.g., 01 02), default to Day-Month
                    d, m = v1, v2
            else:
                groups = match.groups()
                if p_type == 'dMy':
                    d_str, month_name, y_str = groups
                else: # Mdy
                    month_name, d_str, y_str = groups
                
                d, y = int(d_str), int(y_str)
                # Convert month names to numbers
                months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, 
                          "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
                m = months.get(month_name[:3].lower(), 1)
            
            # Logic to handle 2-digit years (e.g., '84' -> 1984, '10' -> 2010)
            y_str = str(y) if 'y_str' not in locals() else y_str
            if len(y_str) <= 2:
                y += 1900 if y > 30 else 2000
                
            return today.year - y - ((today.month, today.day) < (m, d))
            
    return None

def main():
    print("Reggy++: Hello! What is your full name?")
    user_input = input("User: ")
    # [cite_start]Regex to capture the last word as the surname [cite: 32]
    surname_match = re.search(r'\s(\w+)$', user_input.strip())
    surname = surname_match.group(1) if surname_match else user_input
    
    print(f"Reggy++: Nice to meet you, {surname}!")
    print("Reggy++: When were you born?")
    bday_input = input("User: ")
    age = calculate_age(bday_input)
    
    if age is not None:
        print(f"Reggy++: So you are {age} years old!")
    else:
        print("Reggy++: Interesting format! I couldn't quite calculate the age.")

    print("Reggy++: How are you feeling today?")
    mood = input("User: ").lower()
    # [cite_start]Regex handles repetition typos and various synonyms [cite: 31]
    if re.search(r'h+a+p+y|g+o+o+d|f+i+n+e|g+r+e+a+t|a+m+a+z+i+n+g|f+a+n+t+a+s+t+i+c|c+h+e+e+r+f+u+l|j+o+y+f+u+l|r+e+l+a+x+e+d|c+a+l+m', mood):
        print("Reggy++: That's wonderful to hear!")
    elif re.search(r's+a+d|b+a+d|u+n+h+a+p+y|u+p+s+e+t|a+n+g+r+y|f+u+r+i+o+u+s|m+a+d|t+i+r+e+d|e+x+h+a+u+s+t+e+d|s+l+e+e+p+y|a+n+x+i+o+u+s|n+e+r+v+o+u+s|w+o+r+r+i+e+d|b+o+r+e+d|d+u+l+l', mood):
        print("Reggy++: I'm sorry. I hope your day gets better!")
    else:
        print("Reggy++: I understand. Thanks for sharing that with me.")

if __name__ == "__main__":
    main()