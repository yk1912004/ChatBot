import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.chat.util import Chat, reflections

# Define chatbot responses using patterns
pairs = [
    (r"(?i)my name is (.+)", [
        "👋 Hello %1, how can I assist you with your laptop today?",
        "😊 Hi %1! What laptop issue can I help you with?"
    ]),
    (r"(?i)i am (.+)", [
        "😊 Nice to meet you, %1! How can I help with your laptop?",
        "👋 Hi %1! What brings you to our service center today?"
    ]),
    (r"(?i)i'm (.+)", [
        "👋 Hello %1! How can I assist you with your laptop?",
        "😊 Hi %1, do you need help with a laptop issue?"
    ]),
    (r"(?i)hi|hello|hey", [
        "👋 Welcome to the Laptop Service Center – your trusted partner for laptop care! How can we help you today?",
        "😊 Hi there! Thanks for choosing Laptop Service Center. How may we assist you?"
    ]),
    (r"(.*)not turning on(.*)", [
        "⚡ I'm sorry to hear your laptop isn't turning on. Have you tried charging it or checking the power cable?",
        "🔋 If your laptop won't power on, try holding the power button for 10 seconds. If it still doesn't work, we recommend a service visit."
    ]),
    (r"(.*)screen(.*)crack(.*)|broken screen|diplay not working(.*)", [
        "💻 A cracked or broken screen can be replaced at our service center. Would you like to book a repair appointment?",
        "🛠️ We offer screen replacement services for all major laptop brands. Would you like a quote?"
    ]),
    (r"(.*)battery(.*)replace(.*)|battery issue", [
        "🔋 Battery issues are common. We can replace your laptop battery quickly. What is your laptop model?",
        "⚡ If your battery drains fast or doesn't charge, a replacement may be needed. Would you like to schedule a battery replacement?"
    ]),
    (r"(.*)upgrade(.*)ram|memory", [
        "🚀 Upgrading RAM can improve your laptop's performance. Please provide your laptop model for compatibility.",
        "💾 We offer RAM upgrades for most laptops. Would you like to know the options for your device?"
    ]),
    (r"(.*)upgrade(.*)ssd|storage", [
        "⚡ SSD upgrades can make your laptop much faster. Would you like a quote for SSD installation?",
        "💽 We can upgrade your laptop's storage. Please share your laptop model for the best options."
    ]),
    (r"(.*)slow(.*)|performance issue", [
        "🐢 A slow laptop can often be fixed with a tune-up or upgrade. Would you like to book a diagnostic service?",
        "⚙️ Performance issues may be due to software or hardware. We can help identify and fix the problem."
    ]),
    (r"(.*)virus(.*)|malware", [
        "🦠 We offer virus and malware removal services. Would you like to schedule a cleaning?",
        "⚠️ If you suspect a virus, avoid using sensitive accounts and bring your laptop in for a checkup."
    ]),
    (r"(.*)data(.*)recovery|lost files", [
        "💾 Data recovery is possible in many cases. Please do not use the laptop further to avoid overwriting files.",
        "🔍 We provide data recovery services. Would you like to book an appointment?"
    ]),
    (r"(.*)warranty(.*)", [
        "We can check your warranty status if you provide your laptop's serial number.",
        "Warranty coverage varies by brand. Would you like help checking your warranty?"
    ]),
    (r"(.*)service cost|repair cost|price", [
        "Service costs depend on the issue and laptop model. Would you like a free estimate?",
        "We offer transparent pricing. Please describe your laptop issue for a quote."
    ]),
    (r"(.*)appointment|book(.*)service", [
        "You can book a service appointment online or by calling our center. Would you like the booking link?",
        "I'd be happy to help you schedule a service. What date and time work for you?"
    ]),
    (r"(.*)location|where(.*)center", [
        "Our service center is located at Main Street, City Center. Would you like directions?",
        "You can find us at Main Street, near the City Mall. Need a map link?"
    ]),
    (r"(.*)thank you|thanks", [
        "You're welcome! Thank you for choosing Laptop Service Center. We're always here to help.",
        "Glad to assist! Your satisfaction is our priority at Laptop Service Center."
    ]),
    (r"bye|goodbye|exit|quit", [
        "Thank you for contacting Laptop Service Center. We appreciate your trust in us. Have a wonderful day!",
        "Goodbye! We're here whenever you need expert laptop assistance. Thank you for being a valued customer."
    ]),
    (r"(.*)overheating(.*)|hot(.*)", [
        "If your laptop is overheating, make sure the vents are not blocked and clean any dust. Would you like to schedule a cleaning service?",
        "Overheating can cause performance issues. We can check your cooling system at our service center."
    ]),
    (r"(.*)keyboard(.*)not working|keyboard issue", [
        "Keyboard issues can often be fixed by cleaning or replacing the keyboard. Is a specific key not working or the whole keyboard?",
        "We can repair or replace faulty keyboards. Would you like to book a service?"
    ]),
    (r"(.*)wifi(.*)not working|(.*)wifi(.*)issue|(.*)wifi(.*)problem|(.*)internet(.*)issue", [
        "If your WiFi isn't working, try restarting your router and laptop. If the problem persists, we can diagnose hardware or software issues.",
        "Connectivity issues can be frustrating. Would you like to bring your laptop in for a network check?"
    ]),
    (r"(.*)touchpad(.*)not working|mouse issue", [
        "Touchpad problems can be due to driver issues or hardware faults. Have you tried updating your drivers?",
        "We can repair or replace faulty touchpads. Would you like to schedule a repair?"
    ]),
    (r"(.*)audio(.*)not working|sound issue|speaker(.*)problem", [
        "Audio issues can be caused by drivers or hardware. Are you getting no sound at all or is it distorted?",
        "We can diagnose and fix speaker or audio problems. Would you like to book a checkup?"
    ]),
    (r"(.*)bluetooth(.*)not working", [
        "Bluetooth issues can often be fixed by updating drivers. If not, we can help you at our service center.",
        "We can troubleshoot Bluetooth problems for you. Would you like to bring your laptop in?"
    ]),
    (r"(.*)screen(.*)flicker|display issue", [
        "Screen flickering can be caused by loose cables or software issues. We can diagnose and fix display problems.",
        "Display issues are common and often repairable. Would you like to schedule a diagnostic?"
    ]),
    (r"(.*)hinge(.*)broken|hinge issue", [
        "Broken hinges can be repaired or replaced at our service center. Would you like a quote?",
        "We offer hinge repair services for all major laptop brands. Would you like to book a repair?"
    ]),
    (r"(.*)fan(.*)noise|noisy fan", [
        "A noisy fan may indicate dust buildup or a failing component. We can clean or replace the fan for you.",
        "Fan noise is a common issue. Would you like to schedule a cleaning or diagnostic?"
    ]),
    (r"(.*)charging(.*)problem|(.*)not charging|(.*)charging(.*)issue", [
        "Charging problems can be due to faulty adapters, ports, or batteries. Have you tried a different charger?",
        "We can diagnose and fix charging issues. Would you like to bring your laptop in for a check?"
    ]),
    (r"(.*)usb(.*)not working|usb port issue", [
        "USB port issues can be caused by hardware faults or driver problems. Have you tried a different USB device?",
        "We can diagnose and repair faulty USB ports. Would you like to book a service?"
    ]),
    (r"(.*)camera(.*)not working|webcam issue", [
        "Webcam issues may be due to driver problems or hardware faults. Have you checked your camera settings?",
        "We can help fix your laptop's camera. Would you like to schedule a diagnostic?"
    ]),
    (r"(.*)software(.*)install|install(.*)software", [
        "We assist with software installation and troubleshooting. Which software do you need help with?",
        "Software installation support is available. Please specify the software name."
    ]),
    (r"(.*)os(.*)reinstall|reinstall(.*)windows|reinstall(.*)operating system", [
        "We can reinstall your operating system and back up your data if needed. Would you like to book this service?",
        "OS reinstallation can resolve many issues. Do you want to proceed with a fresh install?"
    ]),
    (r"(.*)password(.*)reset|forgot(.*)password", [
        "If you've forgotten your password, we can help reset it securely. Is this for Windows or BIOS?",
        "Password reset services are available. Please specify which password you need to reset."
    ]),
    (r"(.*)trackpad(.*)issue|trackpad(.*)not working", [
        "Trackpad issues can be due to driver or hardware problems. Have you tried updating your drivers?",
        "We can repair or replace faulty trackpads. Would you like to schedule a repair?"
    ]),
    (r"(.*)motherboard(.*)issue|motherboard(.*)problem", [
        "Motherboard issues are serious but often repairable. Would you like a diagnostic to confirm the problem?",
        "We offer motherboard repair and replacement services. Would you like a quote?"
    ]),
    (r"(.*)lan(.*)not working|ethernet(.*)not working", [
        "LAN/Ethernet issues can be due to cable, port, or driver problems. Have you tried a different cable?",
        "We can diagnose and fix LAN port issues. Would you like to bring your laptop in?"
    ]),
    (r"(.*)cd(.*)drive(.*)not working|dvd(.*)drive(.*)not working", [
        "CD/DVD drive issues may be hardware or software related. Do you hear any noise from the drive?",
        "We can repair or replace faulty optical drives. Would you like to book a service?"
    ]),
    (r"(.*)beep(.*)sound|beeping(.*)noise", [
        "Beeping sounds at startup can indicate hardware issues. How many beeps do you hear?",
        "We can diagnose beep codes and fix the underlying problem. Would you like to schedule a diagnostic?"
    ]),
    (r"(.*)screen(.*)black|no display", [
        "A black screen or no display can be caused by hardware or connection issues. Have you tried connecting to an external monitor?",
        "We can diagnose and repair display issues. Would you like to book a checkup?"
    ]),
    (r"(.*)battery(.*)drain|battery(.*)fast", [
        "Rapid battery drain can be due to age or background processes. We can test and replace your battery if needed.",
        "We offer battery diagnostics and replacements. Would you like to schedule a service?"
    ]),
    (r"(.*)(book|fix|schedule|set|arrange)(.*)(appointment|service|visit|meeting)(.*)(on|at)?\s*([0-9]{1,2}(:[0-9]{2})?\s*(am|pm)?|tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday|[0-9]{4}-[0-9]{2}-[0-9]{2})?", [
        "🗓️ Sure! Please confirm the date and time you prefer for your appointment. – Laptop Service Center",
        "📅 I can help you schedule a service. What date and time would you like to book? – Laptop Service Center",
        "👍 Your request to book an appointment has been received. Please specify the exact date and time. – Laptop Service Center"
    ]),
    (r"(.*)change(.*)display(.*)|display(.*)change(.*)|replace(.*)display(.*)", [
        "🖥️ We can help you change or replace your laptop display at Laptop Service Center. Would you like to book a display replacement service?",
        "💡 Display replacement is one of our specialties! Please share your laptop model, and we’ll provide the best options for a new display.",
        "🔧 Need to change your display? Our experts can assist you with a quick and reliable replacement. Would you like to schedule an appointment?"
    ]),
    (r"(.*)touch screen(.*)not working|touchscreen(.*)issue", [
        "🖱️ Touchscreen issues can be caused by hardware or driver problems. Have you tried restarting your laptop or updating drivers?",
        "We can diagnose and repair touchscreen problems. Would you like to bring your laptop in for a checkup?"
    ]),
    (r"(.*)trackpad(.*)sensitive|trackpad(.*)too fast|trackpad(.*)too slow", [
        "Trackpad sensitivity can be adjusted in your laptop settings. Would you like guidance on how to do this, or prefer a technician to check it?",
        "We can help you adjust your trackpad settings or check for hardware issues if needed."
    ]),
    (r"(.*)laptop(.*)shutting down(.*)suddenly|laptop(.*)turns off(.*)", [
        "Sudden shutdowns can be caused by overheating, battery, or hardware issues. Has your laptop been getting hot?",
        "We recommend a diagnostic to find the cause of unexpected shutdowns. Would you like to book a service?"
    ]),
    (r"(.*)charging(.*)slow|slow(.*)charging", [
        "Slow charging can be due to a faulty adapter, battery, or charging port. Have you tried a different charger?",
        "We can test your charging system and recommend a solution. Would you like to bring your laptop in?"
    ]),
    (r"(.*)laptop(.*)noisy|strange(.*)sound|weird(.*)noise", [
        "Unusual noises can come from the fan, hard drive, or speakers. Can you describe the sound?",
        "We can inspect your laptop for any hardware issues causing noise. Would you like to schedule a checkup?"
    ]),
    (r"(.*)laptop(.*)very hot|laptop(.*)heating up(.*)", [
        "If your laptop is getting very hot, it may need cleaning or thermal paste replacement. Would you like to book a cooling system service?",
        "Overheating can damage your laptop. We recommend a professional cleaning. Shall we schedule it?"
    ]),
    (r"(.*)screen(.*)dim|screen(.*)brightness(.*)issue", [
        "Screen brightness issues can be due to settings or hardware. Have you tried adjusting the brightness keys?",
        "We can check your display for hardware faults if brightness adjustment doesn't work. Would you like to book a service?"
    ]),
    (r"(.*)laptop(.*)slow internet|internet(.*)slow", [
        "Slow internet can be caused by network issues or laptop settings. Have you tried restarting your router?",
        "We can help diagnose whether the issue is with your laptop or your network. Would you like assistance?"
    ]),
    (r"(.*)laptop(.*)not booting|won't boot|boot(.*)problem", [
        "Boot problems can be due to software or hardware issues. Do you see any error messages?",
        "We can diagnose and fix boot issues. Would you like to bring your laptop in for a checkup?"
    ]),
    (r"(.*)laptop(.*)beeping(.*)start|beep(.*)startup", [
        "Beeping at startup often indicates a hardware issue. How many beeps do you hear?",
        "We can interpret beep codes and repair the problem. Would you like to schedule a diagnostic?"
    ]),
    (r"(.*)laptop(.*)not detecting(.*)usb|usb(.*)device(.*)not detected", [
        "USB detection issues can be due to port or driver problems. Have you tried other USB devices?",
        "We can diagnose and repair USB port issues. Would you like to book a service?"
    ]),
    (r"(.*)laptop(.*)not detecting(.*)hard drive|hard disk(.*)not detected", [
        "Hard drive detection issues can be serious. Do you see any error messages on startup?",
        "We can check your hard drive connection and health. Would you like to bring your laptop in?"
    ]),
    (r"(.*)laptop(.*)not updating|windows(.*)not updating", [
        "Windows update issues can often be fixed with troubleshooting.5 Would you like step-by-step help or a technician to handle it?",
        "We can resolve update problems for you. Would you like to book a software service?"
    ]),
    (r"(.*)laptop(.*)not connecting(.*)projector|projector(.*)not working", [
        "Projector connection issues can be due to cable, port, or display settings. Have you tried another cable?",
        "We can help configure your laptop for projector use. Would you like to bring it in?"
    ]),
    (r"(.*)", [
        "🤖 This is Laptop Service Center. We're here to help with any laptop service or repair questions. Could you please provide more details?",
        "💬 Let us know your laptop issue or question, and our team at Laptop Service Center will do our best to assist!"
    ])
]

# Update Chatbot logic to handle dict responses
class KeyedChat(Chat):
    def respond(self, str):
        # Overriding to return both key and text
        for (pattern, response_list) in self._pairs:
            match = pattern.match(str)
            if match:
                resp = random.choice(response_list)
                if isinstance(resp, dict):
                    # Replace %1, %2, etc. with matched groups
                    resp_text = resp["text"]
                    for i, group in enumerate(match.groups(), 1):
                        resp_text = resp_text.replace("%%%d" % i, group)
                    return resp["key"], resp_text
                else:
                    # fallback for old string responses
                    resp_text = resp
                    for i, group in enumerate(match.groups(), 1):
                        resp_text = resp_text.replace("%%%d" % i, group if group is not None else "")
                    return "no_key", resp_text
        return "no_key", None

# Initialize chatbot
chatbot = KeyedChat(pairs, reflections)

# Start chat
def start_chat():
    print("Jarvis: Hello! Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "exit", "quit"]:
            print("Jarvis: Goodbye! If you need further assistance, just start a new chat.")
            # Ask for feedback after exit
            feedback = input("Jarvis: Before you go, could you please provide feedback about your experience? ")
            print("Jarvis: Thank you for your feedback!")
            print("Jarvis: We value your feedback and look forward to serving you again. For more updates and offers, visit our website or follow us on social media!")
            break
        key, response = chatbot.respond(user_input)
        if response:
            if key != "no_key":
                print(f"JarvisT [{key}]: {response}")
            else:
                print(f"Jarvis: {response}")
        else:
            print("Jarvis: I'm here to help with any laptop service or repair questions. Could you please provide more details?")

# Run the chatbot
start_chat()
