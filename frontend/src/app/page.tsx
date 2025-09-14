import Chat from '@/components/Chat'
import Evaluations from '@/components/Evaluations'
import Upload from '@/components/Upload'

export default function Home() {
  return (
    <main>
      <div className="row">
        <div className="col"><Upload /></div>
      </div>
      <div className="row mt">
        <div className="col"><Chat /></div>
      </div>
      <div className="row mt">
        <div className="col"><Evaluations /></div>
      </div>
    </main>
  )
}
