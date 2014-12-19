
###############################################################################################
#### Definition des variables globales

places =[]
minX, maxX =0, 0
minY, maxY =0, 0
zminX, zmaxX =0, 0
zminY, zmaxY =0, 0
maxdensity = 0
zoomLevel = 0
zoomcenter = []
black = color(0)
minValPop = 1000
placesel = 0
placesSel = []
maxPop = 0
dep = 0
input= False
inputkonami = False
depinput=[]
depPlaces=[]
konamicode = [UP,UP,DOWN,DOWN,LEFT,RIGHT,LEFT,RIGHT,"b","a"]
konaminum=0



###############################################################################################
#### Setup du projet

def setup():
    global labelFont
    size(1200,800)
    readData()
    noLoop()
    labelFont = loadFont("ArialMT-48.vlw")
    textFont(labelFont, 32)

        
###############################################################################################
#### Fonction de dessin processing 
                                                                                       
def draw():
    global minValPop
    textFont(labelFont, 32)
    background(255)
    for place in places:
        if place.population > minValPop and place.inzoom():
            place.drawp()
            place.drawc()
    #for place in places:
    #   if place.population > minValPop:
    #place.drawc()

    drawSelectedCity()
    fill(255)
    rect(800,0,400,800)
    drawLegend()
    if konaminum == 10:
        textFont(labelFont, 60)
        text("YOU WON", 600,600)
        fill(255,255,0)
        triangle(100, 500, 200, 300, 300, 500)
        triangle(300, 500, 400, 300, 500, 500)
        triangle(200, 300, 300, 100, 400, 300)
        
    




###############################################################################################
#### Definition de la classe place

class Place():
    longitude = 0
    latitude = 0
    name = ""
    postalCode = 0
    population = -1
    density = -1
    #    select = False
    
    def inzoom(self):
        if self.longitude <= zmaxX and self.longitude >= zminX and self.latitude <= zmaxY and self.latitude >= zminY:
            return True
        else:
            return False
    
    # coordonnee x sur la carte
    def x(self):
        return map(self.longitude,zminX, zmaxX,0,width-400)
    
    # coordonnee y sur la carte
    def y(self):
        return map(self.latitude,zminY, zmaxY,height,0)
    
    # alpha sur la densite
    def alphas(self):
        return (self.density)/maxdensity
    
    # methode de dessin des cercles
    def drawc(self):
        fill(0,125)
        noStroke()
        ellipse(self.x(),self.y(),self.population/20000*(zoomLevel+1),self.population/20000*(zoomLevel+1))
    
    # methode de dessin des points
    def drawp(self):
        coloration = (1-self.alphas()*30)*200
        if coloration < 0:
            coloration =0
        
        try:
            fill(coloration)
            noStroke()
            ellipse(self.x(),self.y(),2*(zoomLevel+1),2*(zoomLevel+1))
                
        except:
            pass
            
    # methode pour trouve le point le plus proche        
    def contains(self,a, b, d):
        X = self.x()
        Y = self.y()
        
        if dist(X, Y, a, b) <= d and self.inzoom():
            return True
        else:
            return False
    
    #    def isSelected(self):
    #        return self.select
        
    
    
    
###############################################################################################
#### text fixe

def drawLegend():
    #textFont(labelFont, 32)
    #fill(0)
    #text("Hello, France!", 100, 40)
    fill(0)
    rect(800,0,2,800)
    textFont(labelFont, 16)
    text("Population Minimum = " + str(minValPop),810,720)
    rect(800,700,400,2)
    text("City Name",810,20)
    rect(1000,0,2,700)
    text("Population",1005,20)
    rect(1100,0,2,700)
    text("Density",1105,20)
    rect(1198,0,2,700)
    rect(800,25,400,2)
    stroke(0)
    fill(255)
    rect(1000,750,200,50)
    noStroke()
    fill(0)
    textFont(labelFont, 32)
    fill(255,0,0)
    text("RESET",1050,785)
    
    i=1
    for place in placesSel:
        textFont(labelFont, 16)
        fill(0,0,0)
        text(place.name,805,i*25+20)
        text(str(place.population),1005,i*25+20)
        text(str(place.density),1105,i*25+20)
        rect(800,i*25+25,400,2)
        i+=1
    
    
def drawSelectedCity():
    
    i=1
    for place in placesSel:
        textFont(labelFont, 16)
        texWid = textWidth(place.name)
        if place.population/20000 >= 2:
            stroke(255,0,0)
            fill(255,0,0,125)
            ellipse(place.x(),place.y(),place.population/20000*(zoomLevel+1),place.population/20000*(zoomLevel+1))
        else:
            stroke(0,0,255)
            fill(0,0,255,125)
            ellipse(place.x(),place.y(),2*(zoomLevel+1),2*(zoomLevel+1))
        noStroke()
        fill(255,100)
        rect(place.x()- texWid/2 -5 ,place.y()-25,texWid+10,19)
        fill(255,0,0)
        text(place.name,place.x()- texWid/2, place.y()-10)
        i=i+1
        
    if placesel !=0:
        textFont(labelFont, 16)
        texWid = textWidth(placesel.name)
        if placesel.population/20000 >= 2:
            stroke(0,0,255)
            fill(0,0,255,125)
            ellipse(placesel.x(),placesel.y(),placesel.population/20000*(zoomLevel+1),placesel.population/20000*(zoomLevel+1))
        else:
            stroke(0,0,255)
            fill(0,0,255,125)
            ellipse(placesel.x(),placesel.y(),2*(zoomLevel+1),2*(zoomLevel+1))
        noStroke()
        fill(255,100)
        rect(placesel.x()- texWid/2 -5 ,placesel.y()-25,texWid+10,19)
        fill(0,0,255)
        text(placesel.name,placesel.x()- texWid/2, placesel.y()-10)
        
    

###############################################################################################
#### recherche du point le plus proche

def pick(x,y):
    for place in reversed(places):
        if place.contains(x,y,2) and place.population >=minValPop:
            return place
    return 0

def pickzoom(x,y): 
    for place in reversed(places):
        if place.contains(x,y,10):
            return place
    return 0
        

def removepick(x,y):
    global placesSel
    if x > 800:
        i= y/25 - 1 
        if i in range(0,len(placesSel)):
            placesSel.remove(placesSel[i])

def resetall(x,y):
    global placesSel,zoomLevel,minValPop,zmaxX,zminX,zminY,zmaxY,zoomcenter,konaminum,dep,input,inputkonami
    if x > 1000 and y > 750:
        placesSel = []
        zoomLevel = 0
        zoomcenter = []
        minValPop = 1000
        zminX,zmaxX,zminY,zmaxY = minX,maxX,minY,maxY
        konaminum = 0
        dep = 0
        input= False
        inputkonami = False
        
        
def zoomin(citycenter):
    global zoomcenter,zminX,zmaxX,zminY,zmaxY,zoomLevel
    zoomLevel +=1
    zoomcenter.append(citycenter)
    zminX = citycenter.longitude - (maxX-minX)/2/(zoomLevel+1)
    zmaxX = citycenter.longitude + (maxX-minX)/2/(zoomLevel+1)
    zminY = citycenter.latitude - (maxY-minY)/2/(zoomLevel+1)
    zmaxY = citycenter.latitude + (maxY-minY)/2/(zoomLevel+1)

def zoomout():
    global zoomcenter,zminX,zmaxX,zminY,zmaxY,zoomLevel
    zoomLevel -=1
    zoomcenter.remove(zoomcenter[zoomLevel])
    if zoomLevel == 0:
        zminX = minX
        zmaxX = maxX
        zminY = minY
        zmaxY = maxY
    else:
        citycenter = zoomcenter[zoomLevel - 1]
        zminX = citycenter.longitude - (maxX-minX)/2/(zoomLevel+1)
        zmaxX = citycenter.longitude + (maxX-minX)/2/(zoomLevel+1)
        zminY = citycenter.latitude - (maxY-minY)/2/(zoomLevel+1)
        zmaxY = citycenter.latitude + (maxY-minY)/2/(zoomLevel+1)

def zoomindep():
    global zoomcenter,zminX,zmaxX,zminY,zmaxY,zoomLevel
    print dep , len(depPlaces[dep])
    zoomLevel = 5
    zoomcenter = []
    for i in range(0,5):
        zoomcenter.append(depPlaces[dep][0])
        
    zminX = min(depPlaces[dep], key=lambda place: place.longitude).longitude - (maxX-minX)/2/(zoomLevel+1)
    zmaxX = max(depPlaces[dep], key=lambda place: place.longitude).longitude + (maxX-minX)/2/(zoomLevel+1) 
    zminY = min(depPlaces[dep], key=lambda place: place.latitude).latitude - (maxY-minY)/2/(zoomLevel+1)
    zmaxY = max(depPlaces[dep], key=lambda place: place.latitude).latitude + (maxY-minY)/2/(zoomLevel+1)

###############################################################################################
#### recherche du point le plus proche

def mouseMoved():
    global placesel
    placesel = pick(mouseX,mouseY)
    redraw()


def mouseClicked():
    global placesel,placesSel
    placesel = pick(mouseX,mouseY)
    if placesel !=0:
        if placesel in placesSel:
            placesSel.remove(placesel)
        else:
            placesSel.append(placesel)
    removepick(mouseX,mouseY)
    resetall(mouseX,mouseY)
    redraw()
    
def mouseDragged():
    global zoomcenter,zminX,zmaxX,zminY,zmaxY
    citycenter = pickzoom(mouseX,mouseY)
    if zoomLevel !=0:
        if citycenter != 0:
            zoomcenter.remove(zoomcenter[zoomLevel-1])
            zoomcenter.append(citycenter)
            zminX = citycenter.longitude - (maxX-minX)/2/(zoomLevel+1)
            zmaxX = citycenter.longitude + (maxX-minX)/2/(zoomLevel+1)
            zminY = citycenter.latitude - (maxY-minY)/2/(zoomLevel+1)
            zmaxY = citycenter.latitude + (maxY-minY)/2/(zoomLevel+1)
    redraw()
       
    

    
def keyReleased():
    global minValPop, zoomcenter,zminX,zmaxX,zminY,zmaxY,zoomLevel
    if keyCode == RIGHT and minValPop < maxPop/2:
        minValPop = minValPop * 2
    elif keyCode == LEFT and minValPop >1:
        minValPop = minValPop / 2
    elif keyCode == UP and zoomLevel < 10:
        citycenter = pickzoom(mouseX,mouseY)
        if citycenter != 0:
            zoomin(citycenter)
            
    elif keyCode == DOWN and zoomLevel > 0:
        zoomout()
        
        
    redraw()
    
def keyPressed():
    global dep,input,depinput,konaminum,inputkonami
    if input == True:
        if key in ["0","1","2","3","4","5","6","7","8","9"]:
            depinput.append(key)
            if len(depinput) == 2:
                input = False
        elif key == "e":
            input = False
    if len(depinput) == 2:
        dep = int(depinput[0])*10 + int(depinput[1])*1
        print dep
        zoomindep()
        depinput = []
    if inputkonami == True:
        if keyCode == konamicode[konaminum] or key == konamicode[konaminum]:
            konaminum +=1
            print konaminum
    
    if key == "d":
        input = True
    if key == "k":
        inputkonami = True
    
    

    redraw()
    
             

    

        
###############################################################################################
#### Lecture des donnees
        
def readData():
    global minX,maxX,minY,maxY,maxdensity,zminX,zminY,zmaxX,zmaxY,maxPop,places,depPlaces
    lines = loadStrings("population.tsv")
    #print lines # for debugging
    # First line contains metadata
    # Second line contains column labels
    # Third line and onward contains data cases
    for i in range(0,100):
        depPlaces.append([])
    for line in lines[2:]:
        columns = line.split("\t")
        place = Place()
        place.postalCode = int(columns[0])
        place.longitude = float(columns[1])
        place.latitude = float(columns[2])
        place.name = columns[4]
        place.population = int(columns[5])
        place.density = float(columns[6])
        places.append(place)
        depPlaces[place.postalCode/1000].append(place)

    minX = min(places, key=lambda x: x.longitude).longitude
    maxX = max(places, key=lambda place: place.longitude).longitude
    minY = min(places, key=lambda place: place.latitude).latitude
    maxY = max(places, key=lambda place: place.latitude).latitude
    maxPop = max(places, key=lambda place : place.population).population
    zminX,zmaxX,zminY,zmaxY = minX,maxX,minY,maxY
    maxdensity = max(places, key=lambda place: place.density).density
    print maxdensity
    

